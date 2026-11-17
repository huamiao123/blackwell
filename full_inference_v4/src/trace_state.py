"""Request/iteration state machine, tested without any GPU or vLLM imports."""
from common import classify_schedule,digest,percentile


class TraceState:
    def __init__(self, manifest, start_ns):
        self.manifest=manifest
        self.start_ns=start_ns
        self.requests={}
        self.iterations=[]
        self.replays=[]

    def enqueue(self,rid,prompt,now):
        if rid in self.requests:raise ValueError('duplicate request ID')
        self.requests[rid]=dict(request_id=rid,prompt_id=prompt['prompt_id'],
                               prompt_hash=prompt['token_ids_sha256'],prompt_length=prompt['actual_token_length'],
                               enqueue_ns=now,output_events=[],output_token_ids=[],finished_ns=None)

    def schedule(self,before,scheduled,start,end,preempted=0):
        row=classify_schedule(before,scheduled)
        row.update(scheduler_iteration=len(self.iterations),cpu_start_ns=start,cpu_end_ns=end,
                   preempted=preempted,graph_replay_ids=[],gpu_assignment_status='UNVERIFIED')
        self.iterations.append(row)
        return row

    def output(self,rid,tokens,now):
        if not tokens:return False
        r=self.requests[rid]
        if r['output_events'] and now<r['output_events'][-1]['time_ns']:raise ValueError('nonmonotonic outputs')
        first=not r['output_events']
        r['output_events'].append(dict(time_ns=now,count=len(tokens)))
        r['output_token_ids'].extend(tokens)
        return first

    def finish(self,rid,now):
        if not self.requests[rid]['output_events']:raise ValueError('no output')
        self.requests[rid]['finished_ns']=now

    def summaries(self):
        rows=[]
        for rid,r in self.requests.items():
            es=r['output_events'];n=len(r['output_token_ids'])
            if not es or r['finished_ns'] is None:raise ValueError('incomplete request')
            first,last=es[0]['time_ns'],es[-1]['time_ns']
            gaps=[(y['time_ns']-x['time_ns'])/1e6 for x,y in zip(es,es[1:])]
            single=all(e['count']==1 for e in es)
            rows.append(dict(request_id=rid,prompt_id=r['prompt_id'],prompt_hash=r['prompt_hash'],
                             prompt_length=r['prompt_length'],output_length=n,ttft_ms=(first-r['enqueue_ns'])/1e6,
                             model_request_ms=(r['finished_ns']-r['enqueue_ns'])/1e6,
                             decode_output_span_ms=(last-first)/1e6,
                             tpot_ms=(last-first)/1e6/(n-1) if n>1 else 'NOT_APPLICABLE',
                             itl_p50_ms=percentile(gaps,.5),itl_p95_ms=percentile(gaps,.95),
                             itl_scope='per-token output' if single else 'bundled output events; not token ITL',
                             output_hash=digest(r['output_token_ids']),measurement_scope='offline pretokenized engine',
                             measurement_status='UNVERIFIED'))
        return rows

    def verify_fixed(self,decode_steps):
        for rid,r in self.requests.items():
            if len(r['output_token_ids'])!=decode_steps+1:raise ValueError('wrong output length')
            p=d=0
            for it in self.iterations:
                if it['preempted']:raise ValueError('preemption')
                for q in it['requests']:
                    if q['request_id']==rid:p+=q['prefill_tokens'];d+=q['decode_tokens']
            if p!=r['prompt_length'] or d!=decode_steps:raise ValueError(('scheduled work mismatch',rid,p,d))
        return True
