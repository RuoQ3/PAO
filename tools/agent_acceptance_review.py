"""Apply explicit human review decisions; never infer PASS from transport success."""
from collections import Counter
from pathlib import Path
import argparse
import csv
import json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--directory", required=True)
    args = parser.parse_args()
    directory = Path(args.directory)
    decisions = json.loads((directory / "review_decisions.json").read_text(encoding="utf-8"))
    records = [json.loads(x) for x in (directory / "cases.jsonl").read_text(encoding="utf-8").splitlines()]
    by_id = {r["id"]: r for r in records}
    original = json.loads((directory / "original_cases.json").read_text(encoding="utf-8"))
    results = []
    for number in range(1, 121):
        case_id = f"T{number:03}"
        decision = decisions[case_id]
        evidence_id = decision.get("selected_attempt", case_id)
        row = by_id[evidence_id]
        state = row.get("postconditions", {})
        audits = [e["result"]["audit_id"] for e in state.get("audit_events", []) if e.get("event") == "action_executed"]
        results.append({"id": case_id, "expected": original[case_id],
            "status": decision["status"], "review_reason": decision["reason"],
            "defect_ids": decision.get("defect_ids", []), "selected_attempt": evidence_id,
            "label": row["label"], "input": row["input"], "actual_response": row["response"],
            "preconditions": row.get("preconditions"), "postconditions": state,
            "session_id": row.get("session_id"), "audit_ids": audits,
            "llm_calls": row.get("llm_calls", 0), "elapsed_seconds": row["elapsed_seconds"],
            "evidence": row["evidence"], "additional_evidence": decision.get("additional_evidence", []),
            "original_attempt_transport_status": by_id[case_id]["status"]})
    (directory / "cases_reviewed.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in results), encoding="utf-8")
    with (directory / "cases.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(results[0]))
        writer.writeheader()
        for row in results:
            writer.writerow({k: json.dumps(v, ensure_ascii=False) if isinstance(v, (dict, list)) else v for k, v in row.items()})
    summary = json.loads((directory / "summary.json").read_text(encoding="utf-8"))
    summary["reviewed_unique_case_counts"] = dict(Counter(r["status"] for r in results))
    summary["unresolved_case_ids"] = [r["id"] for r in results if r["status"] in {"FAIL", "BLOCKED", "SKIP"}]
    summary["status"] = "部分完成" if summary["unresolved_case_ids"] else "等待最终完成条件核对"
    (directory / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: summary[k] for k in ("status", "reviewed_unique_case_counts", "unresolved_case_ids")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
