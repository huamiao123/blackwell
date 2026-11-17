# Prompt、Tokenizer 与输入公平性

## 当前数据

已生成 88 条输入：

- 40 条 `synthetic_control`：8 个不同编号记录序列 × 128/512/2048/4096/8192。
- 48 条真实文本输入：8 个不同案例 × 128/256/512/1024/2048/4096。

真实文本的格式覆盖通用阅读问答、代码、长文档、多轮对话，每类 2 个案例。
原文来自公开文学作品和 CPython v3.12.3 源码；对话历史及任务指令为本实验编写，
不是用户真实对话。它是小规模英文格式泛化集，不代表生产请求或全面应用分布。

[source_manifest.json](prompts/sources/source_manifest.json) 保存 URL、Git blob（如适用）、
原始字节 SHA256、编码、许可和本地路径。文学作品保留 Project Gutenberg 通知，
CPython 保留完整源码及 PSF 许可。原站 503 后使用 GITenberg 镜像取得同一公开作品；
镜像版本和字节 Hash 已冻结。Sherlock 源文件明确声明 ISO-8859-1，按该声明解码，
原始字节不改写。

## 模板与合法性

全部使用 BF16 模型目录的规范 Tokenizer 编码，再将完全相同的 ID 交给两模型。
两模型 `tokenizer.json` SHA256 相同、定义集合相同；`tokenizer_config.json` 不同，
EOS 声明也有差异，因此不分别使用默认配置生成两份输入。

固定：`enable_thinking=False`、`add_generation_prompt=True`、`add_special_tokens=False`，
不自动追加 BOS。Special Token 只允许来自固定 Chat Template；逐条消息正文禁止
包含 Special Token。输出固定组为 `ignore_eos=True`、257 输出；自然 EOS 组将另行
固定并记录生效的 EOS/stop ID，当前未执行。

验证内容包括：实际 Tokenizer 定义集合、正文 Special Token、模板重新渲染一致性、
编码后完整 ID 一致性、精确长度、Token Hash、整个 Prompt 文件 Hash、模板与配置
Hash、特殊 Token 位置。不能只用 embedding `vocab_size` 检查。

## 长度与自然性

真实输入采用连续源文片段裁剪并经过完整模板重新编码，搜索精确目标长度；
不重复段落填充。片段可能在句中或代码中结束，指令明确要求承认缺失上下文。
这种裁剪限制已记录，不能当作未经加工的完整原始应用请求。

每条保留消息列表、序列化原文、实际 IDs、长度、来源偏移及 Hash。多轮案例保留
真实的 user/assistant/user 角色结构，历史 assistant 文本是固定的自编输入。

主基线无 System Prompt、Prefix Cache 关闭。Batch 内不同输入去重，计算最长共同
Token 前缀；超过 32 个 Token 拒绝，短模板公共头作为预期脚手架记录。

## 实验匹配

| 组 | 内容 | Batch | 输入长度 |
|---|---|---:|---:|
| E1/E2 | 同一受控来源 | 1/8 | 2048 |
| E3/E4 | 同一真实案例集合，第 0 行用于 B1 | 1/8 | 512 |
| E5/E6 | 同一真实案例集合，第 0 行用于 B1 | 1/8 | 2048 |
| E7 | 固定研究分布 | 8 | 128,256,512,512,1024,2048,2048,4096 |

两精度共享 Manifest 和 Prompt work hash。14 个配置均已准备；执行授权仍只允许
未来经检查的 E1/E2。8192 合成输入仅作为数据准备，没有创建自动执行配置。

上下文检查包括 `prompt_length+257 <= max_model_len`。当前配置统一 4608，E7 的
4096 输入也满足预算；无 chunked prefill 时总输入必须 <=16384。实际 KV 容量与
显存可执行性必须在 Gate 2 由模型初始化和调度记录验收，CPU 检查不作保证。
