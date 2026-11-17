# 真实主 Kernel 的来源与选择

下表自动摘自 Gate 3 表，不手工填入结论。两种精度各 144 个模块、36864 次实际 Decode 投影已在历史 Node Trace 中匹配；那不等于本轮已经重放。

| precision | logical_op | grid_ctas | threads | registers_per_thread | static_shared_bytes | dynamic_shared_bytes |
| --- | --- | --- | --- | --- | --- | --- |
| nvfp4 | qkv_proj | 48 | 384 | 168 | 0 | 88064 |
| nvfp4 | o_proj | 32 | 384 | 168 | 0 | 88064 |
| nvfp4 | gate_up_proj | 192 | 384 | 168 | 0 | 88064 |
| nvfp4 | down_proj | 32 | 384 | 168 | 0 | 88064 |
| bf16 | qkv_proj | 384 | 32 | 124 | 0 | 16896 |
| bf16 | o_proj | 256 | 32 | 124 | 0 | 16896 |
| bf16 | gate_up_proj | 1536 | 32 | 124 | 0 | 16896 |
| bf16 | down_proj | 256 | 32 | 124 | 0 | 16896 |

首轮只选 QKV 和 Gate/Up。固定 layer 18 的原因是中间层、在新测量前冻结，不按延迟挑选。具体模块、历史 Graph/Tools/Node ID、参数 SHA 与来源在 `configs/kernel_selection.json`、`analysis/input_audit.json`。

新进程必须重新捕获并以活权重引用识别同一逻辑节点；Graph ID/Tools ID 与历史指针不能直接复用。不能通过函数名相似、层次顺序或相同 GEMM 形状替代真实绑定。A 的新执行身份与 GPU 时间线相关联之后，才能用于精度/算子并发比较。
