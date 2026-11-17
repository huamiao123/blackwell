# 历史实验审计与继承边界

本文件是新审计，不覆盖旧报告。全部 11 个指定文件已按实际路径读取并记录
SHA256、字节与行数，见 [history_sources.json](analysis/history_sources.json)。
版本基线为仓库提交 `76c150153521c1fc4fe40e7f567a05a88d2d8d2b`。

| 历史内容 | 实际来源 | 审计结论 | V4.1 继承方式 |
|---|---|---|---|
| Graph/Eager Decode 性能与 GPU Metrics | `nsys_round2/final_report.md`、CSV、原脚本 | 原负载下有效；设备级采样与无分析器性能分开 | 继承现象和已验证工具参数，不作为完整推理基线 |
| BF16 B1 GEMV、B8 TensorOp，NVFP4 SM120 E2M1 | `nsys_round2` 的独立 Node Trace | 真实执行路径证据，限定当时的负载和版本 | 作为 Gate 2 需重新核验的预期，不能复制为本轮实测 |
| 新公平 Decode 基线，20 次有效测量 | `resource_overlap_v1` | Prefix Cache 关闭、8 条不同合成输入、真实 Replay；仍是 Decode-only | 保留原计时背景；不与完整推理时间相加或求并发损失 |
| 原并发重叠率与串行基线 | 更早的 `src/interference*`、`audit_v3/source_audit.md` 与 `corrected_results.md` | 部分撤回、部分修正；负载错误、提交偏差与缓存条件影响解释 | 禁止继承旧错误值；新实验使用实测 AB/BA 和真实 Kernel 交集 |
| `resource_overlap_v1` 的并发结果 | `resource_overlap_v1/final_report.md` | **未执行**，不存在可以引用的并发性能或重叠率 | 标记 UNVERIFIED；不能称其为并发失败 |
| Gate 9 合成矩阵实验 | `audit_v3/raw/gate9_M1024.json` 及审计 | 仅聚合中位数，不能恢复完整配对原始样本；平台背景也不同 | 继承公平工作量要求，不据其推断真实完整推理 |
| 辅助 Kernel | `resource_overlap_v1/src/{probes,auxiliary_*}.py` | 代码已准备，GPU 正确性、指令和运行时间没有验收 | 未来逐项 GPU 验收；当前不启用 |
| 旧 Token 输入 | 旧 `prompt_ids()` 与本机 tokenizer JSON | embedding 大小 151936，Tokenizer 定义 151669；旧 B8 有 21 个位置未定义，2 个正文 Special Token | 旧数据保留原背景；V4.1 全量重新编码与校验 |
| 旧结束 Event/计时 | `overlap_runner.py` | 首次 Decode 前全设备同步，从首次 Replay 开始；不能保留为完整推理路径 | 新代码只在完整请求前预热后同步、响应返回后诊断 drain |
| 旧命令汇总缺失 | `resource_overlap_v1/analysis/failures.md` | 汇总文件被旧副本覆盖，独立监督记录完整并已恢复汇总 | 新目录只采用独立监督记录，不回写原历史日志 |

原任务表把“原并发重叠率/旧串行基线”的来源写成 `resource_overlap_v1`，这里按实际
文件纠正。目录名本身不代表已运行或已验证。

## 重要限制

1. 旧 B8 相同 Prompt、Prefix Cache 开启的数据，与后续关闭状态不可混算。
2. 旧 10 kHz 指标是设备级时间采样；不能填入单 Kernel 专属计数栏。
3. Node Trace 与 Graph 级 Metrics 是不同采集，不能伪装为同一次执行。
4. Eager tracing 的明显扰动已经记录，不能由其间隙比例推断正常运行的空闲容量。
5. 旧计数器频率存在负数/不合理值；不取绝对值、不猜单位。新资源分析默认不使用
   未通过单位与范围验收的频率导出。
6. 旧报告的“中断/重填充”机制解释保持假说，不提升为因果事实。
7. 本轮未重试 NCU 注入。安装版本/help 可用，不代表硬件计数可用。

## 当前静态源码核查

[cpu_hook_inspection.json](analysis/cpu_hook_inspection.json) 保留安装源码摘录与文件哈希。
`LLMEngine.add_request()` 可能经 `assign_request_id()` 改写调用方 ID，必须使用其
返回的实际 ID；新 Hook 和 CPU 假 Engine 测试覆盖了这一点。
安装的 UniProcExecutor 即使 `non_block=True`，也在调用线程执行 `run_method()`，
异步返回主要包装输出等待。新框架据此关联 dispatch/replay；真实 GPU 路径仍待
Gate 2 验收，不能由此静态分析宣布映射有效。
