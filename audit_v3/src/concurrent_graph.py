"""Corrected dual-task timing protocol (v3).

Fixes the three fatal flaws of `src/interference_v2.py`:

  A1 CPU submission skew
      -> both tasks are pre-built CUDA Graphs; replay is a single host call each,
         so the host cannot spend tens of milliseconds enqueueing A before B.
         `cpu_submit_ms` is measured and reported so the residual skew is visible.
      -> both submission orders (A->B and B->A) are measured.

  A2 serial baseline is measured, never summed
      -> `run_serial` replays the two graphs back-to-back on ONE stream and times
         that single window.

  A5 alone baseline is re-measured in the same session
      -> `measure_alone` is called by the orchestrator per session, interleaved.

What this does NOT claim:
  * a graph replay is not evidence of simultaneous start, only of a much smaller
    host-side gap;
  * concurrent execution is not evidence of physical kernel overlap - that comes
    from `timeline_analysis.py`;
  * overlap of kernel intervals is not evidence of co-residency on the same SM.
"""
import time
import torch


def build_graph(fn, reps, warm=3, capture_stream=None):
    """Capture `reps` sequential calls of fn into a CUDA Graph."""
    s = capture_stream or torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(warm):
            fn()
    torch.cuda.current_stream().wait_stream(s)
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g, stream=s):
        for _ in range(reps):
            fn()
    torch.cuda.synchronize()
    return g


def _ev():
    return torch.cuda.Event(True)


def measure_alone(graph, rounds=5):
    """Time a single replay of `graph`, median of `rounds`."""
    ts = []
    for _ in range(rounds):
        e0, e1 = _ev(), _ev()
        e0.record(); graph.replay(); e1.record(); torch.cuda.synchronize()
        ts.append(e0.elapsed_time(e1))
    ts.sort()
    return dict(alone_ms=ts[len(ts) // 2], samples_ms=ts)


def timed_eager(fn, reps, warm=2):
    """Time `reps` eager launches. Needed as the *correct* baseline when the
    legacy protocol also used eager launches - mixing a graph-based alone
    baseline with an eager concurrent run is itself a confound."""
    for _ in range(warm):
        fn()
    torch.cuda.synchronize()
    e0, e1 = _ev(), _ev()
    e0.record()
    for _ in range(reps):
        fn()
    e1.record(); torch.cuda.synchronize()
    return e0.elapsed_time(e1)


def run_serial(gA, gB, order="AB"):
    """Both graphs back-to-back on the SAME stream -> a real measured serial baseline."""
    stream = torch.cuda.Stream()
    e0, e1 = _ev(), _ev()
    torch.cuda.synchronize()
    with torch.cuda.stream(stream):
        e0.record()
        if order == "AB":
            gA.replay(); gB.replay()
        else:
            gB.replay(); gA.replay()
        e1.record()
    e1.synchronize()
    return dict(serial_ms=e0.elapsed_time(e1), order=order)


def run_concurrent(gA, gB, order="AB", sA=None, sB=None):
    """Replay both graphs on two streams behind one latch event.

    Returns the measurement plus `cpu_submit_ms` (host time spent issuing the two
    replays) and `start_skew_ms` (device-side start gap between the two streams).

    `sA` / `sB` may be supplied so the caller can keep the same streams across
    captures (needed to identify them later in a GPU timeline).
    """
    sA = sA or torch.cuda.Stream()
    sB = sB or torch.cuda.Stream()
    ctrl = torch.cuda.current_stream()
    ready, a0, a1, b0, b1, joint = _ev(), _ev(), _ev(), _ev(), _ev(), _ev()
    torch.cuda.synchronize()
    ready.record(ctrl)
    sA.wait_event(ready)
    sB.wait_event(ready)

    t0 = time.perf_counter()
    if order == "AB":
        with torch.cuda.stream(sA):
            a0.record(sA); gA.replay(); a1.record(sA)
        with torch.cuda.stream(sB):
            b0.record(sB); gB.replay(); b1.record(sB)
    else:
        with torch.cuda.stream(sB):
            b0.record(sB); gB.replay(); b1.record(sB)
        with torch.cuda.stream(sA):
            a0.record(sA); gA.replay(); a1.record(sA)
    t1 = time.perf_counter()

    ctrl.wait_event(a1)
    ctrl.wait_event(b1)
    joint.record(ctrl)
    joint.synchronize()

    ta, tb = a0.elapsed_time(a1), b0.elapsed_time(b1)
    # device-side start gap: 0 == the two streams started together
    skew = abs(a0.elapsed_time(b0))
    return dict(concurrent_ms=ready.elapsed_time(joint), TA_conc_ms=ta, TB_conc_ms=tb,
                cpu_submit_ms=(t1 - t0) * 1e3, start_skew_ms=skew, order=order)


def run_concurrent_streams(fA, fB, rA, rB, order="AB"):
    """Legacy protocol, kept only to quantify how bad the host skew was.

    Deliberately reimplements the v2 submission pattern; do not use for results.
    """
    sA, sB = torch.cuda.Stream(), torch.cuda.Stream()
    ctrl = torch.cuda.current_stream()
    ready, a0, a1, b0, b1, joint = _ev(), _ev(), _ev(), _ev(), _ev(), _ev()
    torch.cuda.synchronize()
    ready.record(ctrl)
    sA.wait_event(ready); sB.wait_event(ready)
    t0 = time.perf_counter()
    if order == "AB":
        with torch.cuda.stream(sA):
            a0.record(sA)
            for _ in range(rA):
                fA()
            a1.record(sA)
        with torch.cuda.stream(sB):
            b0.record(sB)
            for _ in range(rB):
                fB()
            b1.record(sB)
    else:
        with torch.cuda.stream(sB):
            b0.record(sB)
            for _ in range(rB):
                fB()
            b1.record(sB)
        with torch.cuda.stream(sA):
            a0.record(sA)
            for _ in range(rA):
                fA()
            a1.record(sA)
    t1 = time.perf_counter()
    ctrl.wait_event(a1); ctrl.wait_event(b1)
    joint.record(ctrl); joint.synchronize()
    return dict(concurrent_ms=ready.elapsed_time(joint),
                TA_conc_ms=a0.elapsed_time(a1), TB_conc_ms=b0.elapsed_time(b1),
                cpu_submit_ms=(t1 - t0) * 1e3,
                start_skew_ms=abs(a0.elapsed_time(b0)), order=order)
