# QC-GRDC 本地选择器试验

运行命令：

```powershell
$env:PYTHONPATH = "src"
python scripts/run_qc_grdc_smoke.py
```

协议包含 12 个三轮会话（36 个请求），使用 `WhitespaceTokenizer` 和确定性 oracle backend。oracle 只有在答案字符串仍出现在压缩 prompt 中时才返回正确答案，因此这里衡量的是证据保留能力和选择器开销，不代表 Qwen 7B 的 F1/EM。

| 方法 | Oracle 证据召回 | 平均输入 Token | Token 减少率 | 平均完整请求延迟 | 平均压缩墙钟 |
|---|---:|---:|---:|---:|---:|
| 无压缩 | 100.0% | 172.7 | 0.0% | 14.81 ms | 0.04 ms |
| V2 融合图 | 100.0% | 86.7 | 49.8% | 9.77 ms | 1.87 ms |
| QC-GRDC | 100.0% | 41.7 | 75.8% | 5.56 ms | 1.28 ms |

在这个无模型 pilot 中，QC-GRDC 保持了与无压缩相同的 oracle 证据召回，输入 Token 减少 75.8%，完整请求延迟下降约 62.5%，压缩墙钟比 V2 低约 32%。这是因为达到质量代理下界后会停止继续加入只有重复信息的句子。由于后端延迟是按输入 Token 的确定性模拟，结果只用于验证实现和指标口径；真实质量结论要用同一批多轮序列、同一 7B 模型完成 `evidence_graph_qc_grdc_v4` 对比实验。
