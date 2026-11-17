# Decode注入与请求指标

每次Decode保留Scheduler Iteration、Request ID、实际输入Token/Hash、sequence length、scheduled tokens和Graph replay关联。首输出来自Prefill，之后验证256次实际Decode；不能仅按max_tokens推断。

辅助任务必须有已准备输入和明确依赖Event。精确逐Graph Node注入需要额外活Graph证据，不通过Python forward NVTX猜测每次Replay的层位置。不能精确控制则退回阶段级，报告范围改变。

`request_metrics.py`从单请求相邻输出时间计算ITL，从首末Token间隔除以输出数减1计算TPOT。若只有整批进度、捆绑返回或重复时间戳，则拒绝当作逐Token ITL。没有单请求逐Token事件时保留UNAVAILABLE。

报告逐请求TTFT/TPOT/ITL、Run级均值和P50/P95、辅助工作量与联合时间。256step是一个Run内部序列，不能当256独立实验重复。
