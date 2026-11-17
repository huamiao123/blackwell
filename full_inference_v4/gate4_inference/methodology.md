# 完整推理并发方法

主负载E2：B=8，每条2048个合法合成控制Token，输出257个Token（1 Prefill输出+256 Decode），KV=fp8_e4m3，Graph开启，Prefix Cache/Chunked Prefill关闭。输入与Prompt Hash继承Gate2；这不是自然语言应用泛化或在线服务E2E。

历史P0保持：E1 BF16/NVFP4=2727.1457155/1762.3462699ms，E2=4059.2583297/2143.1284998ms；TPOT E1=10.046077839/6.712654165ms，E2=11.289459325/7.196791254ms。新并发必须在当前条件重测A Alone，不能直接拿历史均值当配对基线。

逐精度比较A Alone、B Alone、Serial AB/BA、Parallel AB/BA、Empty，以及原始推理、普通并发、资源感知并发。全部使用同一请求/输出要求与有效B工作量；串行条件必须真实执行。

Prefill、Decode、Lifecycle、Mixed分别统计。完整请求损失不能掩盖Decode P95恶化；逐请求输出时间才是ITL，调度器一次step的时间不是每请求ITL。离线预编码Engine窗口排除客户端网络和Tokenization，不称为服务端到端窗口。

P0正式性能至少5个配对Block，并进行关键结果独立进程复验。P1指标和P2 Node Timeline分开采，记录扰动且不覆盖P0。联合收益与主任务损失、辅助保持率同时报告。当前没有这些新实验数据。
