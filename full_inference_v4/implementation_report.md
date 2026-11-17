# V4.1 Gate 0/1 执行报告

**本阶段已完成 CPU 侧数据、核心框架、分析契约和安全门禁准备；没有执行模型
推理、CUDA 工作负载、NCU 注入或并发实验。Gate 2 不自动启动。**

## 完成与证据

| 工作 | 已交付 | 验证范围 |
|---|---|---|
| 指定历史审计 | 11份文件逐字节读取、Hash/行数；历史证据映射及错误来源纠正 | CPU 文档/数据审计 |
| 环境与安装工具 | NSYS/NCU version/help、GB20x集合help、安装源码摘录和Hash | 可访问性已核对，硬件采集未验证 |
| Tokenizer修复 | 使用实际定义集合，统一规范Tokenizer、Thinking关闭、模板Special位置记录 | 实际Tokenizer全量重新编码 |
| 输入数据 | 88条Prompt，原文/角色/IDs/长度/来源/文件及Token Hash；保留许可 | 40合成+48真实文本格式输入 |
| 匹配配置 | E1–E7×两精度，共14份；输入、KV、Backend、调度设置匹配 | CPU预算与Hash校验；不保证GPU显存可执行 |
| 完整推理worker | 完整generate、请求ID、逐请求输出/结束、实际调度、Graph Replay Hook | 源码/CPU假Engine契约测试，GPU集成UNVERIFIED |
| 计时 | 无中间全设备同步；CPU请求/调度/GPU区间分开；每请求ITL/TPOT | CPU状态机与负例测试 |
| P0/P1/P2框架 | 无Profiler、Graph+10kMetrics、独立Node；P0至少5次 | 命令构造和默认dry-run；真实扰动待测 |
| Kernel/算子分析 | Schema检查、运行时关联、默认UNKNOWN、Trace绑定映射、融合Kernel并集 | 人工SQLite fixture和独立区间oracle，非新Trace |
| 资源分析 | 设备scope、阶段条件视图、时间加权、缺口/异常频率处理 | CPU测试，所有新硬件值缺失 |
| 并发分析 | 固定工作量校验、实测AB/BA比较、双向指标、B尾部/部分Kernel处理 | CPU数值测试，未运行并发 |
| GPU安全 | 复用空闲/控制器/所有权/进程组/超时；新增窗口、白名单、直接worker门禁 | CPU解析与拒绝路径测试；sudo终止行为待GPU窗口验收 |

具体数据与测试状态见 [quality_validation.json](analysis/quality_validation.json)、
[prompt_validation.json](analysis/prompt_validation.json) 和 [cpu_tests.log](analysis/cpu_tests.log)。
当前为19项CPU单元/契约测试，以及真实Tokenizer的88条正例、5类负例验证。
“VERIFIED”仅表示明确的CPU范围，不扩展到GPU结果。

## 设计已完成但尚未实现或运行的部分

- Teacher Forcing：计划生成器和连续输入Hash检查已实现；sampler强制Token接入、
  Graph兼容和实际消费验证尚未实现，CLI明确禁止execute。
- P3：已有指标与失败边界设计，没有自动NCU启动脚本，不假设注入可用。
- P/D/L/M注入：锚点、GPU实际归属、跨阶段与尾部统计已设计；未接入实际worker。
- Low/Medium/High探针：固定总工作量、改变潜在瞬时并行度的方案已披露；还需要
  GPU Kernel实现/适配、计算结果、编译指令与独立运行验收。
- 动态到达、自然EOS、Prefix Cache专项、在线服务窗口、真实有用消费者尚未执行。
- 论文图绘制保留门禁入口；当前无新GPU结果，不生成虚构图。完整12类图的具体
  展示布局待有效数据后实现，当前不宣称绘图系统已完成。

## CPU阶段遇到并处理的问题

1. NCU默认PATH无可执行文件，改为检查实际安装路径，version/help成功；未重试注入。
2. Gutenberg原站503，记录失败后使用公开GITenberg镜像并冻结文件Hash。
3. Sherlock原文件是ISO-8859-1，按文件声明解码，保留原字节和编码元数据。
4. 读取整个文学作品会触发Tokenizer最大长度警告；改为只编码足够长的候选前缀。
   实际Prompt仍按目标精确长度验证，没有将长源文件送入模型。
5. 安装版vLLM可能重写请求ID；Hook改用返回的实际ID，并增加对应Fake Engine测试。
6. 本地Tokenizer依赖产生pandas/numexpr版本警告，不影响已验证的ID与模板输出；
   没有为此修改系统包。模型运行将使用服务器既有环境。
7. 本地Transformers依赖会间接导入torch，即使设置USE_TORCH=0；因此额外检查其
   CUDA懒初始化标记必须为false。没有调用CUDA可用性查询、分配显存或运行模型。

## 尚未解决、需要GPU授权的事项

1. 验证实际vLLM配置可初始化、KV容量足够，FP4进入SM120原生路径，实际256Decode成立。
2. 验证新Hook没有改变Graph/Fusion/Kernel数量、实际Batch或关键延迟。
3. 验收真实SQLite Schema、GPU身份/时间域、相关ID和Graph Node映射完整性。
4. 测量P1/P2扰动，必要时将映射诊断和低扰动资源采集拆开。
5. 验证sudo NSYS进程树的停止/清理机制；无法清理时立即停本轮，不继续运行。
6. 对重点Kernel补机制证据；NCU不可用时保留缺失，不以编译元数据替代硬件计数。

Gate 0/1的CPU交付可供人工验收；GPU执行有效性仍为UNVERIFIED。
当前没有启用授权窗口、没有自动轮询GPU、没有后台实验，也没有推进Gate 2。

## 下一步人工检查点

先审阅Manifest、输入实例、计时定义及门禁。确认后再给出有起止时间的GPU窗口，
手动运行一个E1/E2配置并检查结果；不一次执行整个Gate链。当前E3–E7配置仅是数据
与预算准备，不代表已经做过显存实测或得到授权。
