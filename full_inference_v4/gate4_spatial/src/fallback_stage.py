"""Adapter for existing Gate-3 hooks. This does not identify an exclusive kernel."""
from contracts import ContractError

class StageFallback:
    def __init__(self,hooks,phase,readiness_provider,submit_probe):
        if phase not in ['PREFILL','DECODE']:raise ContractError('explicit fallback phase required')
        self.hooks=hooks;self.phase=phase;self.ready=readiness_provider;self.submit=submit_probe
        self.events=[];self.original=None
    def install(self):
        executor=self.hooks.core.model_executor;self.original=executor.execute_model
        def wrapped(*args,**kwargs):
            if self.hooks.state is not None and not self.events:
                s=args[0] if args else kwargs.get('scheduler_output')
                row=self.hooks.schedule_ids.get(id(s))
                if row and row['active_request_count'] and row['stage']==self.phase:
                    ready=self.ready(row)
                    if not ready['probe_inputs_ready'] or not ready['dependency_event_recorded']:
                        raise ContractError('probe dependencies not ready')
                    record=dict(iteration=row['scheduler_iteration'],phase=row['stage'],
                        scope='REAL_INFERENCE_STAGE_FALLBACK_NOT_SINGLE_KERNEL',gpu_overlap='UNVERIFIED_UNTIL_P2')
                    self.submit(ready,record);self.events.append(record)
            return self.original(*args,**kwargs)
        executor.execute_model=wrapped
    def restore(self):
        if self.original is not None:self.hooks.core.model_executor.execute_model=self.original
