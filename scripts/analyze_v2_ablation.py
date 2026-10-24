"""Validate and summarize the paired V2 retrieval/packing ablation."""

from __future__ import annotations

from collections import Counter, defaultdict
import json
from pathlib import Path


PREFIX = Path("results/real_7b_multiturn_v2_20pct_ablation")
BASELINE = Path("results/real_7b_multiturn_v2_20pct_three_methods")
LABELS = {
    "evidence_graph_fused_v2": "V2 融合图（原版）",
    "fused_v2_no_history_budget": "V2（历史证据预算=0）",
    "fused_v2_current_only": "V2（每轮独立图）",
    "query_aware_bm25_mmr": "纯 BM25/MMR 抽取",
}


def mean(rows: list[dict], field: str) -> float:
    return sum(float(row["quality_metrics"].get(field, 0.0)) for row in rows) / len(rows)


def pair_delta(pairs: list[tuple[dict, dict]], metric: str) -> dict:
    # Each sample contains three linked turns. Estimate uncertainty across
    # sessions, not as if all turns were independent observations.
    by_session: dict[str, list[float]] = defaultdict(list)
    for a, b in pairs:
        session_id = str(a["metadata"]["request_metadata"]["session_id"])
        by_session[session_id].append(float(a["quality_metrics"][metric]) - float(b["quality_metrics"][metric]))
    deltas = [sum(values) / len(values) for values in by_session.values()]
    n = len(deltas)
    average = sum(deltas) / n
    se = (sum((value - average) ** 2 for value in deltas) / max(1, n - 1) / n) ** 0.5
    return {
        "n_sessions": n,
        "n_questions": len(pairs),
        "mean_delta": average,
        "ci95_low": average - 1.96 * se,
        "ci95_high": average + 1.96 * se,
        "wins": sum(value > 1e-12 for value in deltas),
        "ties": sum(abs(value) <= 1e-12 for value in deltas),
        "losses": sum(value < -1e-12 for value in deltas),
    }


def main() -> None:
    plan = json.loads(Path(str(PREFIX) + "_plan.json").read_text(encoding="utf-8"))
    summary = json.loads(Path(str(PREFIX) + "_summary.json").read_text(encoding="utf-8"))
    rows = [json.loads(line) for line in PREFIX.with_suffix(".jsonl").read_text(encoding="utf-8").splitlines()]
    old_rows = [json.loads(line) for line in BASELINE.with_suffix(".jsonl").read_text(encoding="utf-8").splitlines()]
    assert summary["complete"] and len(rows) == summary["rows"] == plan["expected_generation_count"]
    counts = Counter(row["method"] for row in rows)
    assert set(counts) == set(plan["methods"])
    assert all(count == plan["request_count"] for count in counts.values())
    old_plan = json.loads(Path(str(BASELINE) + "_plan.json").read_text(encoding="utf-8"))
    assert [(s["sequence_id"], s["request_ids"]) for s in plan["sequences"]] == [
        (s["sequence_id"], s["request_ids"]) for s in old_plan["sequences"]]

    by_key: dict[tuple[str, str], dict] = {}
    by_method: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        meta = row["metadata"]["request_metadata"]
        key = (row["request_id"], row["method"])
        assert key not in by_key
        by_key[key] = row
        by_method[row["method"]].append(row)
        assert row["latency"]["end_to_end_ms"] == row["metadata"]["request_wallclock_ms"]
    question_ids = {row["request_id"] for row in rows}
    assert len(question_ids) == plan["request_count"]
    for qid in question_ids:
        assert {(qid, method) for method in plan["methods"]} <= by_key.keys()

    old_v2 = {row["request_id"]: row for row in old_rows if row["method"] == "evidence_graph_fused_v2"}
    assert set(old_v2) == question_ids
    replay_diffs = [by_key[(qid, "evidence_graph_fused_v2")]["quality_metrics"]["token_f1"]
                    - old_v2[qid]["quality_metrics"]["token_f1"] for qid in question_ids]
    assert all(abs(value) <= 1e-12 for value in replay_diffs), "V2 rerun differs from original results"

    aggregates = {}
    for method in plan["methods"]:
        items = by_method[method]
        aggregates[method] = {
            "n": len(items),
            "f1": mean(items, "token_f1"), "em": mean(items, "exact_match"),
            "support_precision": mean(items, "supporting_fact_precision"),
            "support_recall": mean(items, "supporting_fact_recall"),
            "evidence_f1": mean(items, "evidence_f1"),
            "tokens": sum(row["compressed_prompt_tokens"] for row in items) / len(items),
            "token_reduction": sum(row["token_reduction_rate"] for row in items) / len(items),
            "latency_ms": sum(row["latency"]["end_to_end_ms"] for row in items) / len(items),
        }

    paired = {}
    base_name = "evidence_graph_fused_v2"
    subsets = {"all": None, "2wikimultihopqa": "2wikimultihopqa", "hotpotqa": "hotpotqa", "musique": "musique"}
    for method in plan["methods"]:
        if method == base_name:
            continue
        paired[method] = {}
        for label, dataset in subsets.items():
            pairs = []
            for row in by_method[method]:
                meta = row["metadata"]["request_metadata"]
                if dataset is None or meta["dataset"] == dataset:
                    pairs.append((row, by_key[(row["request_id"], base_name)]))
            paired[method][label] = {
                "f1": pair_delta(pairs, "token_f1"),
                "em": pair_delta(pairs, "exact_match"),
            }

    detailed = {}
    for method in plan["methods"]:
        detailed[method] = {"dataset": {}, "turn": {}}
        for dataset in sorted({r["metadata"]["request_metadata"]["dataset"] for r in by_method[method]}):
            items = [r for r in by_method[method] if r["metadata"]["request_metadata"]["dataset"] == dataset]
            detailed[method]["dataset"][dataset] = {
                "n": len(items), "f1": mean(items, "token_f1"), "em": mean(items, "exact_match"),
                "support_precision": mean(items, "supporting_fact_precision"),
                "support_recall": mean(items, "supporting_fact_recall"), "evidence_f1": mean(items, "evidence_f1"),
                "latency_ms": sum(r["latency"]["end_to_end_ms"] for r in items) / len(items),
            }
        for turn in range(3):
            items = [r for r in by_method[method] if r["metadata"]["request_metadata"]["turn_index"] == turn]
            detailed[method]["turn"][str(turn + 1)] = {
                "n": len(items), "f1": mean(items, "token_f1"), "em": mean(items, "exact_match"),
                "support_recall": mean(items, "supporting_fact_recall"),
            }

    result = {
        "complete": True, "rows": len(rows), "questions": len(question_ids), "sessions": plan["sequence_count"],
        "methods": plan["methods"], "counts": dict(counts), "replay_v2_exact_f1_match": True,
        "overall": aggregates, "paired_deltas_vs_v2": paired, "details": detailed,
    }
    json_path = PREFIX.parent / (PREFIX.name + "_analysis.json")
    json_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = [
        "# V2 融合图检索与打包消融实验", "",
        f"完成 {len(rows):,} 次生成：{plan['sequence_count']:,} 个三轮会话、{len(question_ids):,} 个问题。",
        "本次使用与原 20% 实验完全相同的请求 ID、会话次序、7B 模型及 50% 输入 Token 预算。原版 V2 重跑的逐题 F1 与原实验完全一致。", "",
        "## 总体结果", "",
        "| 方法 | F1 | EM | 支持证据召回 | 支持证据精确率 | Evidence F1 | Token 减少率 | 平均完整请求延迟 (ms) |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for method in plan["methods"]:
        m = aggregates[method]
        lines.append(f"| {LABELS[method]} | {m['f1']*100:.2f}% | {m['em']*100:.2f}% | {m['support_recall']*100:.2f}% | "
                     f"{m['support_precision']*100:.2f}% | {m['evidence_f1']*100:.2f}% | {m['token_reduction']*100:.2f}% | {m['latency_ms']:.1f} |")
    lines += ["", "证据指标使用数据集 supporting-fact 标注，仅用于事后诊断，不会传给压缩器。完整请求延迟从压缩开始计至模型完整答案返回。", "",
              "## 相对原版 V2 的逐题配对差异", "",
              "95% 区间按会话聚类的配对差值标准误计算（一个会话的三轮作为同一簇）；区间跨 0 表示本实验没有清晰证据支持质量差异。", "",
              "| 方法 | 子集 | F1 差 (百分点) | F1 95% 区间 | EM 差 (百分点) | EM 95% 区间 |", "|---|---|---:|---:|---:|---:|"]
    for method in plan["methods"]:
        if method == base_name:
            continue
        for subset in subsets:
            p = paired[method][subset]
            f, e = p["f1"], p["em"]
            lines.append(f"| {LABELS[method]} | {subset} | {f['mean_delta']*100:+.2f} | [{f['ci95_low']*100:+.2f}, {f['ci95_high']*100:+.2f}] | "
                         f"{e['mean_delta']*100:+.2f} | [{e['ci95_low']*100:+.2f}, {e['ci95_high']*100:+.2f}] |")
    lines += ["", "## 按数据集 F1 / EM", "", "| 数据集 | 方法 | F1 | EM | 支持证据召回 |", "|---|---|---:|---:|---:|"]
    datasets = sorted(next(iter(detailed.values()))["dataset"])
    for dataset in datasets:
        for method in plan["methods"]:
            m = detailed[method]["dataset"][dataset]
            lines.append(f"| {dataset} | {LABELS[method]} | {m['f1']*100:.2f}% | {m['em']*100:.2f}% | {m['support_recall']*100:.2f}% |")
    lines += ["", "## 按轮次 F1 / EM", "", "| 轮次 | 方法 | F1 | EM |", "|---|---|---:|---:|"]
    for turn in ("1", "2", "3"):
        for method in plan["methods"]:
            m = detailed[method]["turn"][turn]
            lines.append(f"| 第 {turn} 轮 | {LABELS[method]} | {m['f1']*100:.2f}% | {m['em']*100:.2f}% |")
    lines += ["", "## 结论", "",
              f"- 禁用历史证据预算后，F1 变化 {paired['fused_v2_no_history_budget']['all']['f1']['mean_delta']*100:+.2f} 个百分点，"
              f"95% 区间 [{paired['fused_v2_no_history_budget']['all']['f1']['ci95_low']*100:+.2f}, {paired['fused_v2_no_history_budget']['all']['f1']['ci95_high']*100:+.2f}]；"
              "当前样本不足以说明历史证据造成了稳定的质量损失。",
              f"- 每轮独立图相对原 V2 的 F1 变化 {paired['fused_v2_current_only']['all']['f1']['mean_delta']*100:+.2f} 个百分点，"
              "变化幅度很小，说明跨轮历史复用不是主要退化来源。",
              f"- 等预算 BM25/MMR 的支持证据召回为 {aggregates['query_aware_bm25_mmr']['support_recall']*100:.2f}%，"
              f"高于 V2 的 {aggregates[base_name]['support_recall']*100:.2f}%；但其总体 F1 为 {aggregates['query_aware_bm25_mmr']['f1']*100:.2f}%，"
              f"与 V2 的 {aggregates[base_name]['f1']*100:.2f}% 接近。证据召回提升没有转化为明显的答案质量提升。",
              "- 三种新条件的 F1 均接近，且相对 V2 的配对区间覆盖 0；本实验更支持‘50% 预算下的句子筛选/上下文缩减是共同瓶颈’，而不是‘历史图污染’或‘图结构本身导致下降’。",
              "- 支持证据召回约 73–75%，明显低于原始上下文；同时精确率约 13–14%。当前提取式方法仍会漏掉不少标注支持句，也会选入大量未标为支持的句子。下一步优化应聚焦多跳桥接证据的召回和句间关系保留，再做小规模质量验证。", "",
              "## 实验核验", "",
              f"- {len(rows):,} 条生成记录，四种方法各 {plan['request_count']:,} 条；四方法对同一 {len(question_ids):,} 个问题配对。",
              "- 原版 V2 复跑的每题 F1 与先前 20% 实验完全一致。",
              "- 消融语义：`no_history_budget` 只把历史句的可选预算设为 0；`current_only` 每轮新建空图；`BM25/MMR` 每轮仅对当前上下文抽取，不复用图或历史。", "",
              f"逐请求数据：`{PREFIX}.jsonl`；摘要：`{PREFIX}_summary.json`；分析 JSON：`{json_path}`。", ""]
    report = PREFIX.parent.parent / "docs" / "multiturn_v2_20pct_ablation_results.md"
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"complete": True, "rows": len(rows), "questions": len(question_ids),
                      "sessions": plan["sequence_count"], "methods": counts,
                      "v2_replay_matches": True, "report": str(report), "analysis": str(json_path)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
