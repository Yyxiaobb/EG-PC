"""Verify a completed V2 run and write paired session-level comparisons."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from hashlib import sha256
import json
from pathlib import Path
import random
from statistics import mean


LABELS = {
    "no_compression": "无压缩",
    "evidence_graph_dual_path_v1": "旧双路图",
    "llmlingua2": "请求前 LLMLingua-2",
    "evidence_graph_fused_v2": "V2 融合图",
    "evidence_graph_fused_post_llmlingua2_v2": "V2 图 + 生成后 LLMLingua-2",
}
COMBO = "evidence_graph_fused_post_llmlingua2_v2"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prefix", type=Path, default=Path("results/real_7b_multiturn_v2_3pct"))
    parser.add_argument("--bootstrap-replicates", type=int, default=5000)
    parser.add_argument("--report", type=Path, default=Path("docs/multiturn_v2_3pct_results.md"))
    args = parser.parse_args()
    prefix = args.prefix
    plan = json.loads(Path(str(prefix) + "_plan.json").read_text(encoding="utf-8"))
    summary = json.loads(Path(str(prefix) + "_summary.json").read_text(encoding="utf-8"))
    rows = [json.loads(x) for x in prefix.with_suffix(".jsonl").read_text(encoding="utf-8").splitlines()]
    assert summary["complete"] and len(rows) == summary["rows"] == plan["expected_generation_count"]
    assert all(sha256(Path(p).read_bytes()).hexdigest() == h for p, h in plan["code_sha256"].items())
    by_block = defaultdict(list)
    by_question = defaultdict(dict)
    for row in rows:
        meta = row["metadata"]["request_metadata"]
        by_block[(meta["session_id"], row["method"])].append(row)
        key = (meta["session_id"], meta["turn_index"], row["request_id"])
        assert row["method"] not in by_question[key]
        by_question[key][row["method"]] = row
        cm = row["metadata"]["compression_metadata"]
        assert not cm.get("history_compression_error")
        if row["method"] in ("llmlingua2", "evidence_graph_fused_v2", COMBO):
            assert row["compressed_prompt_tokens"] <= cm["budget_tokens"]
        assert row["latency"]["end_to_end_ms"] == row["metadata"]["request_wallclock_ms"]
        if row["method"] == COMBO:
            assert not cm["history_compression_pending"]
    for spec in plan["sequences"]:
        for method in plan["methods"]:
            block = by_block[(spec["sequence_id"], method)]
            assert [r["request_id"] for r in block] == spec["request_ids"]
            assert [r["metadata"]["request_metadata"]["turn_index"] for r in block] == [0, 1, 2]
    for group in by_question.values():
        assert set(group) == set(plan["methods"])
        # The full baseline prompt supplies the exact question without loading
        # datasets or allowing reference annotations into runtime decisions.
        suffix = "\nQuestion:" + group["no_compression"]["metadata"]["compressed_prompt"].rsplit("\nQuestion:", 1)[1]
        for method in ("llmlingua2", "evidence_graph_fused_v2", COMBO):
            assert group[method]["metadata"]["compressed_prompt"].endswith(suffix)
    assert len(by_question) == plan["request_count"]
    assert len(by_block) == plan["sequence_count"] * len(plan["methods"])

    rng = random.Random(20261015)
    pairs = []
    for baseline in plan["methods"]:
        if baseline == COMBO:
            continue
        for metric, extract, scale in (
            ("token_f1_pp", lambda r: r["quality_metrics"]["token_f1"], 100),
            ("exact_match_pp", lambda r: r["quality_metrics"]["exact_match"], 100),
            ("end_to_end_ms", lambda r: r["latency"]["end_to_end_ms"], 1),
        ):
            diffs = [mean(extract(r) for r in by_block[(s["sequence_id"], COMBO)])
                     - mean(extract(r) for r in by_block[(s["sequence_id"], baseline)])
                     for s in plan["sequences"]]
            n = len(diffs)
            samples = sorted(sum(diffs[rng.randrange(n)] for _ in range(n)) / n
                             for _ in range(args.bootstrap_replicates))
            pairs.append({"method": COMBO, "baseline": baseline, "metric": metric,
                          "difference": mean(diffs) * scale,
                          "ci95": [samples[int(len(samples) * .025)] * scale,
                                   samples[min(len(samples) - 1, int(len(samples) * .975))] * scale]})

    combo_rows = [r for r in rows if r["method"] == COMBO]
    first_turn_pairs = [g for key, g in by_question.items() if key[1] == 0]
    adjacent = [r["metadata"]["request_metadata"]["adjacent_relation"] for r in rows
                if r["method"] == "no_compression" and r["metadata"]["request_metadata"]["turn_index"] > 0]
    background_distributions = {}
    for key in ("history_compression_ms", "history_queue_ms", "history_persist_ms", "history_total_ms"):
        values = sorted(r["metadata"]["compression_metadata"][key] for r in combo_rows)
        background_distributions[key] = {"mean": mean(values), "p50": values[round((len(values)-1)*.5)],
                                         "p95": values[round((len(values)-1)*.95)], "max": max(values)}
    diagnostics = {
        "verified_rows": len(rows), "verified_sessions": plan["sequence_count"],
        "verified_questions": len(by_question), "source_hashes_verified": True,
        "history_errors": 0, "budget_violations_v2_and_online": 0,
        "complete_question_checks_passed": True,
        "method_rows": dict(Counter(r["method"] for r in rows)),
        "combo_requests_using_compressed_memory": sum(r["metadata"]["compression_metadata"].get("compressed_memory_selected_count", 0) > 0 for r in combo_rows),
        "first_turn_v2_vs_combo_identical_prompts": sum(g[COMBO]["metadata"]["compressed_prompt"] == g["evidence_graph_fused_v2"]["metadata"]["compressed_prompt"] for g in first_turn_pairs),
        "first_turn_v2_vs_combo_identical_outputs": sum(g[COMBO]["metadata"]["output_text"] == g["evidence_graph_fused_v2"]["metadata"]["output_text"] for g in first_turn_pairs),
        "generation_at_32_token_limit": dict(Counter(r["method"] for r in rows if r["output_tokens"] >= plan["max_new_tokens"])),
        "max_prompt_tokens": max(r["original_prompt_tokens"] for r in rows),
        "adjacent_pairs": len(adjacent),
        "adjacent_with_shared_query_entities": sum(bool(r.get("shared_query_entities")) for r in adjacent),
        "adjacent_with_shared_context_titles": sum(bool(r.get("shared_titles")) for r in adjacent),
        "background_distributions_ms": background_distributions,
        "max_cached_prompt_tokens": max(r["metadata"]["inference_metadata"]["raw_timings"].get("cache_n", 0) for r in rows),
    }
    analysis_path = Path(str(prefix) + "_analysis.json")
    analysis_path.write_text(json.dumps({"diagnostics": diagnostics, "paired_cluster_bootstrap": pairs,
        "bootstrap_unit": "three-turn session", "replicates": args.bootstrap_replicates,
        "difference_direction": "combo minus baseline", "seed": 20261015}, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = ["# V2 三数据集约 3% 真实多轮实验结果", "",
        f"已完成 {len(rows)} 次真实生成，{plan['sequence_count']} 个独立三轮会话、{plan['request_count']} 个不同问题。",
        "使用本地 Qwen2.5-7B-Instruct Q4_K_M 和真实 LLMLingua-2；每种方法每个会话重新建图。", "",
        "HotpotQA 222 / 7405，2Wiki 378 / 12576，MuSiQue 72 / 2417。采用标题/实体关联构造的合成三轮问答，不能等同自然多轮对话。", "",
        "## 总体结果", "",
        "| 方法 | F1 % | EM % | 平均输入 token | 平均压缩 ms | 平均完整延迟 ms | P50 ms | P95 ms |",
        "|---|---:|---:|---:|---:|---:|---:|---:|"]
    overall = {s["method"]: s for s in summary["summaries"] if s["group"] == "overall"}
    for method in plan["methods"]:
        s = overall[method]
        lines.append(f"| {LABELS[method]} | {s['token_f1']*100:.2f} | {s['exact_match']*100:.2f} | {s['input_tokens']:.1f} | {s['compression_wallclock_ms']:.1f} | {s['end_to_end_wallclock_ms']:.1f} | {s['p50_end_to_end_ms']:.1f} | {s['p95_end_to_end_ms']:.1f} |")
    for group, title in (("dataset", "分数据集"), ("turn", "分轮次")):
        lines += ["", f"## {title}", "", "| 分组 | 方法 | n | F1 % | EM % | 平均完整延迟 ms |", "|---|---|---:|---:|---:|---:|"]
        for s in summary["summaries"]:
            if s["group"] == group:
                lines.append(f"| {s['value']} | {LABELS[s['method']]} | {s['n']} | {s['token_f1']*100:.2f} | {s['exact_match']*100:.2f} | {s['end_to_end_wallclock_ms']:.1f} |")
    s = overall[COMBO]
    lines += ["", "## 后台任务与证据复用", "",
        f"组合方法后台模型压缩平均 {s['history_compute_ms']:.1f} ms，队列平均 {s['history_queue_ms']:.1f} ms，提交到入图平均 {s['history_total_ms']:.1f} ms。",
        f"1000 ms 配置空窗下，下轮开始前就绪 {s['history_ready_numerator']} / {s['history_ready_denominator']}（{s['history_ready_rate']*100:.2f}%）；最终轮排除，后台错误 0。",
        f"压缩来源片段累计入选 {s['compressed_memory_selected_count']} 次，涉及 {diagnostics['combo_requests_using_compressed_memory']} / {len(combo_rows)} 个请求。补充本轮未选来源累计 {s['history_supplemented_source_count']} 条次。",
        "后台时间不计入已返回答案的延迟，但不是零成本；它与下一轮的硬件竞争已经体现在请求墙钟中。", "",
        "| 后台阶段 | 平均 ms | P50 ms | P95 ms | 最大 ms |",
        "|---|---:|---:|---:|---:|"]
    for key, d in background_distributions.items():
        lines.append(f"| {key} | {d['mean']:.1f} | {d['p50']:.1f} | {d['p95']:.1f} | {d['max']:.1f} |")
    lines += ["",
        "| 方法 | 平均 prefill ms | 平均 decode ms | 旧式分项之和 ms | 平均历史条数 | 分支 |",
        "|---|---:|---:|---:|---:|---|"]
    for method in plan["methods"]:
        s = overall[method]
        lines.append(f"| {LABELS[method]} | {s['prefill_ms']:.1f} | {s['decode_ms']:.1f} | {s['component_sum_ms']:.1f} | {s['reused_evidence_count']:.2f} | {json.dumps(s['retrieval_winners'], ensure_ascii=False)} |")
    lines += ["", "## 配对比较的不确定性", "",
        f"以独立三轮会话为重采样单位，固定种子进行 {args.bootstrap_replicates} 次配对 bootstrap。差值为组合方法减对照；质量单位为百分点，延迟单位为 ms。区间不代表自然用户总体的外推保证。", "",
        "| 对照 | 指标 | 差值 | 95% 区间 |", "|---|---|---:|---|"]
    for p in pairs:
        lines.append(f"| {LABELS[p['baseline']]} | {p['metric']} | {p['difference']:+.2f} | [{p['ci95'][0]:+.2f}, {p['ci95'][1]:+.2f}] |")
    lines += ["", "## 验证与边界", "",
        "- 验证了行数、每会话三轮顺序、跨方法问题一致、当前代码与计划哈希一致、后台无错误、V2/在线预算无超限、完整问题保留。",
        f"- 首轮无历史时 V2/组合提示相同 {diagnostics['first_turn_v2_vs_combo_identical_prompts']} / {plan['sequence_count']}，输出相同 {diagnostics['first_turn_v2_vs_combo_identical_outputs']} / {plan['sequence_count']}。",
        "- 所有压缩方法使用共同 50% 原生提示 token 上限；实际长度不同，不能宣称严格等长。",
        f"- 相邻问题共 {len(adjacent)} 对，其中共享提问实体 {diagnostics['adjacent_with_shared_query_entities']} 对、共享材料标题 {diagnostics['adjacent_with_shared_context_titles']} 对。共享材料可能包含干扰文档，不保证每一对都构成语义上的追问。",
        "- 原始证据仍保留为溯源锚点，压缩记忆是补充；这不是纯压缩图存储实验。",
        "- 旧双路对照保留旧抢答策略，但采用共同来源 ID 修正及预算，与旧 10% 实验的配置不完全相同。",
        "- LLMLingua-2 关闭整段筛选以保留片段来源映射；与过去启用默认整段筛选的基线有差异。",
        "- 最多生成 32 token、确定性解码、单台 GPU、单次测量；长答案截断、GPU 调度与热状态会影响质量和延迟。",
        "- 各方法达到 32 token 输出上限的请求数：" + json.dumps(diagnostics["generation_at_32_token_limit"], ensure_ascii=False) + "。",
        f"- 服务端缓存 token 的观测最大值为 {diagnostics['max_cached_prompt_tokens']}，原始提示最长 {diagnostics['max_prompt_tokens']} token。",
        "- 完整延迟从客户端压缩开始到生成结果返回，包含请求开销，排除初始化、评估及后处理；不是首 token 延迟。",
        "- 仅当质量与延迟同时获得支持时才能声称兼顾二者，后台工作移出当前请求不自动等于端到端收益。", "",
        "完整协议见 [multiturn_v2_protocol.md](multiturn_v2_protocol.md)。结果汇总与清单位于 `results/real_7b_multiturn_v2_3pct_*`，逐请求 `.jsonl` 与服务日志保存在本地。", ""]
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps(diagnostics, ensure_ascii=False, indent=2))
    print(f"Report: {args.report}; analysis: {analysis_path}")


if __name__ == "__main__":
    main()
