# QC-GRDC：质量约束图条件率失真编码

QC-GRDC（`evidence_graph_qc_grdc_v4`）是 V2 融合图的算法版本。它把每个候选证据句看作一个带来源的编码符号：句子的成本是目标模型 tokenizer 的 Token 数，句子的收益是它为当前问题增加的可观测证据特征。

对候选集合 (C)，选择集合 (Z) 的目标写成：

\[
\min_Z \sum_{s\in Z} c(s)
\quad\text{s.t.}\quad
\frac{Q(Z)}{Q(C)} \ge \rho,
\quad c(Z)\le B.
\]

这里 (B) 是请求预算，\(\rho\) 是 `quality_floor`（默认 0.92），\(Q\) 是运行时质量代理，不使用 supporting-facts 或答案标签。代理由以下可观测量构成：

- 问题中的实体、数字、日期和否定词是否被当前证据覆盖；
- 检索图中的桥接实体是否仍有路径；
- 当前来源是否有至少一个句子保留；
- 句子来源 ID、路径和文档多样性是否可追踪。

算法先保护质量核心：数字/日期/否定词和当前来源的最短高分句优先进入集合。之后反复计算

\[
\frac{\Delta Q(s\mid Z)}{c(s)}
\]

选择边际质量增益除以 Token 成本最大的句子，直到达到质量下界或预算耗尽。来源和句子 ID 只参与选择及审计，不被当作答案质量特征计数，避免重复段落让质量约束无法满足。若单句超过预算，最后才进行 tokenizer 感知截断，并保留问题和 `Answer:` 分隔符。

该方法的算法创新点是把证据图的多跳路径完整性和答案变量覆盖放入率失真约束，而不是先压缩再事后观察质量。它仍是一个无训练、无额外 LLM 调用的选择器，因此压缩开销主要是图搜索、特征集合和 tokenizer 计数，适合放在请求前关键路径；生成后的历史持久化继续由现有异步 hook 完成。

## 接入方式

`SessionEvidenceGraphCompressor` 可以按会话创建独立图：

```python
from ladc.compressors import (
    QualityConstrainedGraphRateDistortionCompressor,
    SessionEvidenceGraphCompressor,
)

compressor = SessionEvidenceGraphCompressor(
    compressor_class=QualityConstrainedGraphRateDistortionCompressor,
    compressor_kwargs={"quality_floor": 0.92, "max_hops": 3},
)
```

真实多轮入口已注册方法名 `evidence_graph_qc_grdc_v4`。例如只比较无压缩、LLMLingua-2 和 QC-GRDC：

```powershell
python -u scripts/run_real_7b_multiturn_v2.py `
  --methods no_compression llmlingua2 evidence_graph_qc_grdc_v4 `
  --fraction 0.01 `
  --output-prefix results/qc_grdc_1pct
```

结果应同时报告 Token 减少率、各轮 F1/EM、压缩墙钟耗时、prefill/decode、完整请求墙钟延迟、质量代理比率、桥接路径保留率和来源回退率。`quality_proxy_ratio` 是算法约束的运行时下界，不能替代真实 7B 模型的 F1/EM；最终结论必须以同一批多轮序列上的真实生成实验为准。

## 当前验证

新增的 3 个 QC-GRDC 单元测试覆盖高失真数字/否定词、会话历史中的多跳桥接和同图同预算的确定性。与现有测试合计 76 项通过（跳过仓库中既存的缺失脚本收集项）。真实 7B 实验依赖本机模型权重和数据集，代码已接入但不会在缺少权重时伪造质量结果。
