# LADC: Latency-Aware Dynamic Prompt Compression

本项目面向大语言模型推理加速，研究在质量允许小幅下降的约束下，如何根据请求特征动态选择 prompt 压缩预算，从而降低输入 token、端到端延迟和单请求成本。

## 研究目标

给定原始 prompt `x`，系统选择压缩策略 `m` 和压缩预算 `b`，使：

```text
compression_overhead + compressed_inference_latency
< original_inference_latency
```

同时控制回答质量下降不超过预设阈值。

## 第一阶段范围

- 本地推理模型：优先适配 Qwen2.5-7B-Instruct GGUF 或同规模模型。
- 压缩策略：无压缩、固定预算、截断、BM25、EAMMR、动态预算。
- 主要工作负载：长上下文问答和 RAG 风格 prompt。
- 主要指标：prefill latency、decode latency、compression latency、端到端 P50/P95、输入 token、吞吐和质量下降。
- 研究重点：latency break-even point，而不是单纯追求压缩率或 QA SOTA。

## 目录

```text
configs/       实验配置
scripts/       实验入口
src/ladc/      核心代码
tests/         单元测试
results/       实验汇总、分析和清单；逐请求 JSONL 和日志保留在本地
docs/          实验协议和研究记录
```

## 研究约束

- 本项目与旧工作区独立，禁止直接修改旧项目文件。
- 所有 token 数量优先使用目标生成模型的原生 tokenizer 统计。
- 所有延迟报告必须拆分压缩、prefill、decode 和端到端部分。
- 只有在压缩总耗时低于推理节省时，才计为真实加速。

## LLMLingua-2 小规模实验

`LLMLingua2Compressor` 使用本地 LLMLingua-2 token-classification 模型，
`scripts/run_llmlingua2_fraction.py` 会从 HotpotQA、2WikiMultihopQA 和 MuSiQue
各抽取指定比例的数据，并比较无压缩、固定预算、证据图和 LLMLingua-2：

```powershell
python scripts/run_llmlingua2_fraction.py --fraction 0.10
```

LLMLingua-2 的模型初始化时间不计入单条压缩延迟；每条请求的推理压缩时间会写入结果。

## 会话级双路证据检索

会话证据图使用 `DualPathEvidenceGraphCompressor` 并行检索会话历史图和当前请求的原始材料。首个通过最低证据门槛的分支用于当前回答，另一分支继续完成并记录检索审计；当前请求的原始证据仍会写入本会话图，供后续问题使用。不同 `session_id` 之间的证据图彼此隔离。

## 安装与测试

需要 Python 3.10 或更新版本：

```powershell
python -m pip install -e ".[dev]"
python -m pytest -q
```

真实数据实验还需要 `llmlingua`、`transformers`、`datasets` 和匹配硬件的 PyTorch。
模型权重、tokenizer、数据集及 llama.cpp 运行库需单独准备，不包含在仓库中。
`configs/local.yaml` 和各脚本的默认路径来自本地实验环境，运行前请修改配置或传入路径参数。

## 多轮实验与结果

`scripts/run_real_7b_multiturn_fraction.py` 比较无压缩、双路会话证据图、
在线 LLMLingua-2、证据图加生成后 LLMLingua-2 四种方法。例如：

```powershell
python scripts/run_real_7b_multiturn_fraction.py --fraction 0.10 --sequence-length 3 --inter-turn-wait-ms 1000
```

用 `--help` 查看模型、tokenizer、数据集和运行库的路径参数。
本轮结果保存在 `results/real_7b_multiturn_10pct_summary.json`：共 746 个三轮会话、
2238 个问题，四种方法合计 8952 次生成。会话按可见材料中的标题和实体关联构造，
并非随机抽取的自然对话；实际覆盖 HotpotQA 741 题、2Wiki 1257 题、MuSiQue 240 题。
每种方法、每个会话分别维护状态。

当前实现和实验仍有以下限制：

- 双路证据图自身采用检索、排序和预算选择；LLMLingua-2 用于独立在线基线及组合方法的生成后记忆压缩。
- 组合方法仍保存原始证据，压缩记忆作为补充；后台压缩未必能在下一轮前完成。
- 汇总中的 `end_to_end_ms` 是在线压缩、服务端 prefill 和 decode 的耗时之和，不等于客户端完整墙钟延迟；后台压缩另行统计。
- 方法执行顺序固定且共享硬件，延迟可能受到缓存与后台任务竞争影响。
- 较早的 `real_7b_post_generation_combo_10pct_summary.json` 实际是 18 个问题的试验，文件名中的 10pct 不代表完整的 10% 抽样。

逐请求输出和服务日志仍保留在本地，Git 仅跟踪 JSON 汇总、分析和实验清单。
