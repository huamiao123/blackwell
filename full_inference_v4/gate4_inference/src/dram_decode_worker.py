"""One accepted streamed read at Decode replays 33–64 of real E2 inference."""
from pathlib import Path
import sys

from dram_probe_gpu import StreamProbe, SYMBOL
from gate44_worker import Gate44Hooks, V4, main


class DramHooks(Gate44Hooks):
    def prepare_probe(self):
        import json
        admission = json.loads((V4/'gate4_inference/analysis/dram_20261119/acceptance.json').read_bytes())
        record, = [r for r in admission['rows'] if r['symbol']==SYMBOL]
        if not record['direction_detected']:
            raise ValueError('standalone physical read signal not verified')
        self.streaming=StreamProbe(self.torch)
        self.work=self.streaming.select(SYMBOL)
        self.cubin_sha=self.streaming.sha
        if self.cubin_sha!=admission['cubin_sha256'] or admission['uuid']!=str(self.streaming.device.uuid).removeprefix('GPU-'):
            raise ValueError('acceptance and live binary/UUID differ')
        self.stream=self.streaming.stream
        poisoned=self.torch.from_numpy((~self.streaming.digest).view('int32')).to('cuda')
        self.buffers=poisoned.expand(len(self.conditions),len(self.targets),-1).clone()
        return poisoned.clone()

    def launch(self,output):
        self.streaming.launch(output)

    def verify_output(self,output):
        return self.streaming.validate(output)

    def validate_alone(self,output):
        return self.streaming.alone(output)


if __name__=='__main__':
    sys.argv.extend(['--phase','decode'])
    main(DramHooks,Path(__file__).resolve())
