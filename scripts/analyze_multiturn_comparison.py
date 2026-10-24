"""Validate a selected-method V2 run and report compression and QA metrics."""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path


LABELS = {
    "no_compression": "无压缩",
    "llmlingua2": "LLMLingua-2",
    "evidence_graph_fused_v2": "V2 融合图",
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prefix", type=Path, default=Path("results/real_7b_multiturn_v2_20pct_three_methods"))
    parser.add_argument("--report", type=Path, default=Path("docs/multiturn_v2_20pct_three_methods_results.md"))
    args = parser.parse_args()
    prefix = args.prefix
    plan = json.loads(Path(str(prefix) + "_plan.json").read_text(encoding="utf-8"))
    summary = json.loads(Path(str(prefix) + "_summary.json").read_text(encoding="utf-8"))
    rows = [json.loads(line) for line in prefix.with_suffix(".jsonl").read_text(encoding="utf-8").splitlines()]
    assert summary["complete"] and len(rows) == summary["rows"] == plan["expected_generation_count"]
    assert len(set(plan["methods"])) == len(plan["methods"])
    assert all(sha256(Path(path).read_bytes()).hexdigest() == digest
               for path, digest in plan["code_sha256"].items())
    grouped = {}
    questions = {}
    by_question = {}
    for row in rows:
        meta = row["metadata"]["request_metadata"]
        cm = row["metadata"]["compression_metadata"]
        grouped.setdefault((meta["session_id"], row["method"]), []).append(row)
        question_key = (meta["session_id"], meta["turn_index"], row["request_id"])
        questions.setdefault(question_key, set()).add(row["method"])
        by_question.setdefault(question_key, {})[row["method"]] = row
        assert row["method"] in plan["methods"]
        assert row["latency"]["end_to_end_ms"] == row["metadata"]["request_wallclock_ms"]
        if row["method"] != "no_compression":
            assert row["compressed_prompt_tokens"] <= cm["budget_tokens"]
            assert "\nQuestion:" in row["metadata"]["compressed_prompt"]
    for sequence in plan["sequences"]:
        for method in plan["methods"]:
            block = grouped[(sequence["sequence_id"], method)]
            assert [r["request_id"] for r in block] == sequence["request_ids"]
            assert [r["metadata"]["request_metadata"]["turn_index"] for r in block] == [0, 1, 2]
    assert len(questions) == plan["request_count"]
    assert all(methods == set(plan["methods"]) for methods in questions.values())
    for group in by_question.values():
        suffix = "\nQuestion:" + group["no_compression"]["metadata"]["compressed_prompt"].rsplit("\nQuestion:", 1)[1]
        for method in ("llmlingua2", "evidence_graph_fused_v2"):
            assert group[method]["metadata"]["compressed_prompt"].endswith(suffix)

    overall = {s["method"]: s for s in summary["summaries"] if s["group"] == "overall"}
    turns = {(s["method"], s["value"]): s for s in summary["summaries"] if s["group"] == "turn"}
    datasets = {(s["method"], s["value"]): s for s in summary["summaries"] if s["group"] == "dataset"}
    lines = [
        "# 三数据集 20% 多轮实验：三种方法对比", "",
        f"完成 {len(rows):,} 次真实生成，{plan['request_count']:,} 个问题，{plan['sequence_count']:,} 个独立三轮会话。",
        "按数据集覆盖：" + "；".join(f"{k} {v['used_in_multiturn_sessions']}/{v['target_questions']} 题" for k, v in plan["coverage"].items()) + "。",
        "会话类型：" + json.dumps({kind: sum(s.get("kind") == kind for s in plan["sequences"])
                                   for kind in sorted({s.get("kind", "unknown") for s in plan["sequences"]})}, ensure_ascii=False) + "。",
        "问题序列按数据集可见标题和实体关联构造；这是合成相关多轮问答，不代表自然用户对话。",
        "模型为本地 Qwen2.5-7B-Instruct Q4_K_M；压缩方法使用相同 50% 原生提示 token 上限，具体输入长度仍以实测为准。", "",
        "## 总体指标", "",
        "| 方法 | Token 减少率 | 第 1 轮回答 F1 | 第 2 轮回答 F1 | 第 3 轮回答 F1 | F1 | EM | 平均完整请求延迟 (ms) |",
        "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for method in plan["methods"]:
        s = overall[method]
        per_turn = [turns[(method, str(i))]["token_f1"] * 100 for i in (1, 2, 3)]
        lines.append(f"| {LABELS.get(method, method)} | {s['token_reduction_rate']*100:.2f}% | "
                     f"{per_turn[0]:.2f}% | {per_turn[1]:.2f}% | {per_turn[2]:.2f}% | "
                     f"{s['token_f1']*100:.2f}% | {s['exact_match']*100:.2f}% | {s['end_to_end_wallclock_ms']:.1f} |")
    lines += ["", "Token 减少率按每条请求相对未压缩提示词的实际 token 减少比例计算，再对请求取平均；无压缩为 0%。F1/EM 为每题得分的平均值。完整请求延迟从客户端压缩开始，到完整生成返回为止，包含压缩及请求/推理路径耗时，不包含初始化和评分。", "",
        "## 按轮次 F1", "", "| 轮次 | 方法 | F1 | 平均完整请求延迟 (ms) |", "|---|---|---:|---:|"]
    for turn in ("1", "2", "3"):
        for method in plan["methods"]:
            s = turns[(method, turn)]
            lines.append(f"| 第 {turn} 轮 | {LABELS.get(method, method)} | {s['token_f1']*100:.2f}% | {s['end_to_end_wallclock_ms']:.1f} |")
    lines += ["", "## 按数据集", "", "| 数据集 | 方法 | n | Token 减少率 | F1 | EM | 平均完整请求延迟 (ms) |", "|---|---|---:|---:|---:|---:|---:|"]
    for dataset in sorted({v for m, v in datasets}):
        for method in plan["methods"]:
            s = datasets[(method, dataset)]
            lines.append(f"| {dataset} | {LABELS.get(method, method)} | {s['n']} | {s['token_reduction_rate']*100:.2f}% | {s['token_f1']*100:.2f}% | {s['exact_match']*100:.2f}% | {s['end_to_end_wallclock_ms']:.1f} |")
    lines += ["", "## 实验核验", "",
        f"- 样本覆盖：{plan['sequence_count']} 个隔离的三轮会话，{len(questions)} 个跨方法相同的问题。",
        f"- 每种方法 {plan['request_count']} 次生成；总计 {len(rows)} 次；代码哈希及每轮顺序校验通过。",
        "- LLMLingua-2 与 V2 融合图提示均通过预算及问题完整性检查。",
        "- 所有方法使用同一批问题序列；各方法、各会话独立建图。", "",
        f"实验计划与逐请求汇总保存在 `{prefix}_plan.json`、`{prefix}_summary.json`；逐请求 JSONL 和服务日志保存在本地。", ""]
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"complete": summary["complete"], "rows": len(rows),
                      "sessions": plan["sequence_count"], "questions": len(questions),
                      "methods": plan["methods"], "code_hashes_verified": True}, ensure_ascii=False, indent=2))
    print(f"Report: {args.report}")


if __name__ == "__main__":
    main()
