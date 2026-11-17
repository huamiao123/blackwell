from pathlib import Path
import sys,unittest
from types import SimpleNamespace as NS
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from bf16_observer import BF16Observer

class FakeGraph:
    def raw_cuda_graph(self):return 3
    def get_graph_data(self):
        return dict(exec_graph_id=3,nodes=[dict(node_id=2,graph_id=3,tools_id=(3<<32)|2,dependencies=[])])

class ObserverContractTests(unittest.TestCase):
    def test_capture_bind_replay_schema_without_cuda(self):
        o=BF16Observer.__new__(BF16Observer)
        o.snapshots=[];o.live=[];o.replays=[]
        o.runtime=NS(cudaGraphNodeType=NS(cudaGraphNodeTypeKernel=0),
            cudaGraphGetNodes=lambda raw,numNodes:(0,([1] if numNodes else None),1),
            cudaGraphNodeGetType=lambda node:(0,0),cudaGraphNodeGetToolsId=lambda node:(0,(3<<32)|2))
        params=NS(func=1,gridDimX=8,gridDimY=256,gridDimZ=1,blockDimX=32,blockDimY=1,blockDimZ=1,
                  sharedMemBytes=16896,kernelParams=0,extra=1)
        o.driver=NS(CUgraphNode=lambda init_value:init_value,cuGraphKernelNodeGetParams=lambda node:(0,params),
                    cuFuncGetName=lambda f:(0,b'cutlass_wmma_bf16'),cuFuncGetParamInfo=lambda f,i:(0,0,16))
        graph=FakeGraph();o.observe(graph);o.bind([]);o.replay(graph,0)
        self.assertEqual(len(o.snapshots[0]['matrix_kernels']),1)
        self.assertEqual(o.snapshots[0]['matrix_kernels'][0]['binding']['status'],'NO_EXACT_REFERENCE')
        self.assertEqual(o.replays,[dict(iteration=0,capture_index=0,exec_graph_id=3)])

if __name__=='__main__':unittest.main()
