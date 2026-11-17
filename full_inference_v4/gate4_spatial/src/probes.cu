// Offline-compilable SM120 probes. No host main, module load, or kernel launch.
#include <cuda_bf16.h>
#include <mma.h>
#include <stdint.h>
using namespace nvcuda;

__device__ __forceinline__ float real_fma(float x) {
  float y;
  asm volatile("fma.rn.f32 %0, %1, %2, %3;" : "=f"(y)
               : "f"(x), "f"(0x1.000002p0f), "f"(0x1p-20f));
  return y;
}

template<int THREADS, int CHUNK, bool STAGED>
__device__ __forceinline__ void fma_body(const float* input, float* output,
                                        uint32_t items, uint32_t rounds) {
  extern __shared__ float staging[];
  constexpr uint32_t TILE = 512, VEC = 16;
  uint32_t tiles = (items + TILE - 1) / TILE;
  for (uint32_t tile = blockIdx.x; tile < tiles; tile += gridDim.x) {
    if constexpr (STAGED) {
      for (uint32_t j=threadIdx.x;j<TILE*VEC;j+=THREADS) {
        uint32_t at=tile*TILE*VEC+j;
        staging[j]=at<items*VEC?input[at]:0.0f;
      }
      __syncthreads();
    }
    for (uint32_t i=threadIdx.x;i<TILE;i+=THREADS) {
      uint32_t at=tile*TILE+i;
      if (at>=items) continue;
      #pragma unroll 1
      for (uint32_t chunk=0;chunk<VEC;chunk+=CHUNK) {
        float acc[CHUNK];
        #pragma unroll
        for(int j=0;j<CHUNK;++j)
          acc[j]=STAGED?staging[i*VEC+chunk+j]:input[at*VEC+chunk+j];
        #pragma unroll 1
        for(uint32_t r=0;r<rounds;++r) {
          #pragma unroll
          for(int j=0;j<CHUNK;++j) acc[j]=real_fma(acc[j]);
        }
        #pragma unroll
        for(int j=0;j<CHUNK;++j) output[at*VEC+chunk+j]=acc[j];
      }
    }
    if constexpr (STAGED) __syncthreads();
  }
}

extern "C" __global__ __launch_bounds__(32)
void fp32_light(const float* input,float* output,uint32_t items,uint32_t rounds) {
  fma_body<32,4,false>(input,output,items,rounds);
}
extern "C" __global__ __launch_bounds__(128)
void fp32_heavy(const float* input,float* output,uint32_t items,uint32_t rounds) {
  fma_body<128,16,true>(input,output,items,rounds);
}

template<int WARPS,int KTILE>
__device__ __forceinline__ void tensor_body(const __nv_bfloat16* input_a,
    const __nv_bfloat16* input_b,float* output,uint32_t tiles,uint32_t rounds) {
  extern __shared__ __align__(16) unsigned char storage[];
  __nv_bfloat16* mem=reinterpret_cast<__nv_bfloat16*>(storage);
  constexpr int WIDTH=16, HALF=WIDTH*KTILE;
  int warp=threadIdx.x/32;
  uint32_t groups=(tiles+WARPS-1)/WARPS;
  for(uint32_t group=blockIdx.x;group<groups;group+=gridDim.x) {
    // Every staged element is a real A/B operand used by the warp's MMA loop.
    for(int j=threadIdx.x;j<WARPS*2*HALF;j+=WARPS*32) {
      int local=j%(2*HALF);
      mem[j]=local<HALF?input_a[local]:input_b[local-HALF];
    }
    __syncthreads();
    uint32_t tile=group*WARPS+warp;
    if(tile<tiles) {
      auto* a=mem+warp*2*HALF;
      auto* b=a+HALF;
      wmma::fragment<wmma::matrix_a,16,16,16,__nv_bfloat16,wmma::row_major> af;
      wmma::fragment<wmma::matrix_b,16,16,16,__nv_bfloat16,wmma::col_major> bf;
      wmma::fragment<wmma::accumulator,16,16,16,float> cf;
      wmma::fill_fragment(cf,0.0f);
      #pragma unroll 1
      for(uint32_t r=0;r<rounds;++r) {
        int k=(r%(KTILE/16))*16;
        wmma::load_matrix_sync(af,a+k,KTILE);
        wmma::load_matrix_sync(bf,b+k,KTILE);
        wmma::mma_sync(cf,af,bf,cf);
      }
      wmma::store_matrix_sync(output+tile*256,cf,16,wmma::mem_row_major);
    }
    __syncthreads();
  }
}

extern "C" __global__ __launch_bounds__(32)
void tensor_light(const __nv_bfloat16* a,const __nv_bfloat16* b,float* output,uint32_t tiles,uint32_t rounds) {
  tensor_body<1,16>(a,b,output,tiles,rounds);
}
extern "C" __global__ __launch_bounds__(128)
void tensor_heavy(const __nv_bfloat16* a,const __nv_bfloat16* b,float* output,uint32_t tiles,uint32_t rounds) {
  tensor_body<4,128>(a,b,output,tiles,rounds);
}
