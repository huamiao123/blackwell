"""Stage-level integration contract. Exact operator entry requires separate proof."""
from useful_work import ContractError

class StageInjection:
    def __init__(self,phase,submit_callback):
        if phase not in ['PREFILL','DECODE','LIFECYCLE','MIXED']:raise ContractError('phase')
        self.phase=phase;self.submit=submit_callback;self.record=None

    def on_dispatch(self,row,readiness):
        if self.record is not None:return False
        actual=row['stage']
        if actual!=self.phase and self.phase!='LIFECYCLE':return False
        if not readiness['inputs_ready'] or not readiness['dependency_event_recorded']:
            raise ContractError('GPU prerequisite not recorded/ready')
        if actual=='MIXED' and not (row['scheduled_prefill_tokens'] and row['scheduled_decode_tokens']):
            raise ContractError('incorrect mixed iteration label')
        self.record=dict(iteration=row['scheduler_iteration'],phase=actual,
                         prefill_tokens=row['scheduled_prefill_tokens'],decode_tokens=row['scheduled_decode_tokens'],
                         gpu_entry_status='UNVERIFIED_UNTIL_TIMELINE',scope='STAGE_LEVEL_NOT_A_TARGET_KERNEL')
        self.submit(readiness['dependency_event'],self.record)
        return True
