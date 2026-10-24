# MuSiQue V3 abstraction pilot

The pilot stopped after 12 matched questions. The intended 2,417-question full comparison was not completed because the abstractive method failed its grounding check in every matched case.

| 方法 | n | F1 | EM | Token减少率 | 平均端到端延迟 | 平均压缩耗时 |
|---|---:|---:|---:|---:|---:|---:|
| no_compression | 12 | 0.0965 | 0.0000 | 0.0% | 1335 ms | 5 ms |
| llmlingua2 | 12 | 0.1556 | 0.0000 | 50.1% | 1367 ms | 637 ms |
| evidence_graph_fused_v2 | 12 | 0.0897 | 0.0000 | 50.3% | 1098 ms | 65 ms |
| evidence_graph_abstractive_v3 | 12 | 0.0897 | 0.0000 | 50.3% | 8903 ms | 7874 ms |

V3 grounding fallback: 12/12 (100%); local summarizer mean latency: 7833 ms; mean generated tokens: 361.

In the matched pilot, V3 used the V2 prompt after rejection for every question. Its F1 and token reduction consequently equal V2, while adding about 7.9 seconds of synchronous compression time and raising mean end-to-end latency to about 8.9 seconds. The local 0.5B model often emitted document titles instead of source-grounded relation notes, so these numbers are a failure diagnosis, not a valid estimate of a functioning abstractive compressor. No quality conclusion should be drawn from n=12.

Raw checkpoint: `results\musique_full_hop_abstractive_comparison_v2.jsonl`
Plan: `results/musique_full_hop_abstractive_comparison_v2_plan.json`
Summary JSON: `results/musique_full_hop_abstractive_comparison_v2_pilot_summary.json`
