// Independent SM120 diagnostic. One pass over an integral number of tiles.
#include <stdint.h>

constexpr unsigned tile_words = 8192;

enum class Operation { Read, Write, Copy };

__device__ __forceinline__ uint32_t output_word(uint32_t index, uint32_t seed) {
  uint32_t value = (index ^ seed) * 0x9e3779b9u;
  return value ^ (value >> 16);
}

template<Operation Mode, unsigned Threads>
__device__ __forceinline__ void stream_words(const uint32_t* source,
    uint32_t* destination, uint32_t tiles, uint32_t seed) {
  extern __shared__ uint32_t staging[];
  __shared__ uint32_t warp_partials[Threads / 32];
  constexpr bool staged = Threads == 128;
  for (uint32_t tile = blockIdx.x; tile < tiles; tile += gridDim.x) {
    const uint32_t base = tile * tile_words;
    uint32_t sum = 0;
    if constexpr (staged) {
      for (unsigned i = threadIdx.x; i < tile_words; i += Threads) {
        if constexpr (Mode == Operation::Write)
          staging[i] = output_word(base + i, seed);
        else
          staging[i] = __ldcg(source + base + i);
      }
      __syncthreads();
    }
    for (unsigned i = threadIdx.x; i < tile_words; i += Threads) {
      const unsigned index = staged ? i ^ 32u : i;
      uint32_t word;
      if constexpr (staged)
        word = staging[index];
      else if constexpr (Mode == Operation::Write)
        word = output_word(base + index, seed);
      else
        word = __ldcg(source + base + index);
      if constexpr (Mode == Operation::Read)
        sum += word * ((base + index) | 1u);
      else
        destination[base + index] = word;
    }
    if constexpr (Mode == Operation::Read) {
      for (unsigned stride = 16; stride; stride >>= 1)
        sum += __shfl_down_sync(0xffffffff, sum, stride);
      if (threadIdx.x % 32 == 0)
        warp_partials[threadIdx.x / 32] = sum;
      __syncthreads();
      if (threadIdx.x == 0) {
        uint32_t digest = 0;
        for (unsigned warp = 0; warp < Threads / 32; ++warp)
          digest += warp_partials[warp];
        destination[tile] = digest;
      }
    }
    if constexpr (staged || Mode == Operation::Read)
      __syncthreads();
  }
}

#define STREAM_ENTRY(NAME, MODE, THREADS)                              \
  extern "C" __global__ __launch_bounds__(THREADS)                     \
  void NAME(const uint32_t* source, uint32_t* destination,             \
            uint32_t tiles, uint32_t seed) {                          \
    stream_words<Operation::MODE, THREADS>(source, destination, tiles, seed); \
  }

STREAM_ENTRY(dram_read_light, Read, 32)
STREAM_ENTRY(dram_read_heavy, Read, 128)
STREAM_ENTRY(dram_write_light, Write, 32)
STREAM_ENTRY(dram_write_heavy, Write, 128)
STREAM_ENTRY(dram_copy_light, Copy, 32)
STREAM_ENTRY(dram_copy_heavy, Copy, 128)
