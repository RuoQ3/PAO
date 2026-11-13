"""Export current live evidence without rerunning LLM or Aspen."""
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
import argparse
import csv
import hashlib
import json
import shutil
import sqlite3
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str), encoding="utf-8")


def csv_write(path, rows):
    if not rows:
        return
    fields = list(dict.fromkeys(k for row in rows for k in row))
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows({k: json.dumps(v, ensure_ascii=False) if isinstance(v, (dict, list)) else v
                          for k, v in row.items()} for row in rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--directory", required=True)
    args = parser.parse_args()
    directory = Path(args.directory).resolve()
    records = [json.loads(x) for x in (directory / "cases.jsonl").read_text(encoding="utf-8").splitlines()]
    # Extract only original execution turns: later read-only turns contain the
    # same audit events and must not replace original duration or approval IO.
    executions = {}
    for case in ("R-T111", "R-T112", "R-T113", "R2-T114"):
        row = json.loads((directory / "sessions" / f"{case}.json").read_text(encoding="utf-8"))
        for event in row["postconditions"]["audit_events"]:
            if event.get("event") != "action_executed":
                continue
            aid = event["result"]["audit_id"]
            executions[aid] = {**event, "action_wall_seconds": row["elapsed_seconds"], "source_case": case}
            write(directory / "executions" / f"{aid}.json", executions[aid])
            write(directory / "executions" / f"{aid}-gateway.json", row["gateway_audits"])

    conn = sqlite3.connect(directory / "simulation.db")
    conn.row_factory = sqlite3.Row
    tables = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    evaluation_rows = []
    for table in sorted(tables):
        rows = [dict(r) for r in conn.execute('SELECT * FROM "' + table.replace('"', '""') + '"')]
        if table in {"execution_evaluations", "execution_budget"}:
            csv_write(directory / f"{table}.csv", rows)
        if table == "execution_evaluations":
            evaluation_rows = rows
    conn.close()
    action_rows, observations, cleanup = [], [], []
    for aid, event in executions.items():
        result = event["result"]
        summary = result.get("summary", {})
        action_rows.append({"audit_id": aid, "source_case": event["source_case"],
            "status": result.get("status"), "algorithm": result.get("algorithm"),
            "evaluations_used": result["evaluations_used"], "stop_reason": summary.get("stop_reason", result.get("stop_reason")),
            "status_counts": summary.get("status_counts", result.get("execution_evidence", {}).get("counts")),
            "conservative_wall_seconds_including_analysis": event["action_wall_seconds"]})
        for index, entry in enumerate(summary.get("records", []), 1):
            obs = entry["observation"]
            observations.append({"audit_id": aid, "evaluation": index, "candidate_id": obs["candidate_id"],
                "generation": entry["candidate"]["generation"], "inputs": entry["inputs"],
                "status": obs["status"], "simulation_valid": obs["status"] in {"feasible", "infeasible"},
                "objectives": obs["objectives"], "constraint_residuals_le_zero": obs["residuals"],
                "evidence_id": obs["evidence_id"]})
        for index, case in enumerate(result.get("simulation_cases", []), 1):
            outputs = case.get("impact_outputs", {})
            if not summary.get("records"):
                observations.append({"audit_id": aid, "evaluation": index, "candidate_id": case["case_id"],
                "inputs": case.get("impact_inputs", case.get("design_vars")), "status": case["status"],
                "simulation_valid": case["simulation_valid"], "outputs": {k: v for k, v in outputs.items() if not k.startswith("_")},
                "constraint_residuals_le_zero": None, "constraint_note": "本动作读取标量，未作优化约束判定",
                "run_id": case.get("run_id"), "run_time": case.get("run_time"), "error": case.get("sim_result", {}).get("error")})
            if outputs.get("_lifecycle"):
                cleanup.append({"audit_id": aid, "case_id": case["case_id"], **outputs["_lifecycle"]})
        cleanup.append({"audit_id": aid, "scope": "action", **result.get("resources", {})})
        write(directory / "executions" / f"{aid}-evidence.json", result.get("execution_evidence", {}))
    csv_write(directory / "actions.csv", action_rows)
    csv_write(directory / "observations.csv", observations)
    write(directory / "cleanup.json", cleanup)
    baseline = json.loads((directory / "baseline.json").read_text(encoding="utf-8"))
    hashes = [{"path": name, "before": expected,
               "after": hashlib.sha256(Path(name).read_bytes()).hexdigest()}
              for name, expected in baseline["source_hashes"].items()]
    for item in hashes:
        item["unchanged"] = item["before"] == item["after"]
    write(directory / "source_hash_verification.json", hashes)
    logs = [json.loads(x) for x in (directory / "llm_calls.jsonl").read_text(encoding="utf-8").splitlines()]
    covered = sorted({r["id"] for r in records if len(r["id"]) == 4 and r["id"].startswith("T")})
    # Conservative: count only successful user turns from the instrumented driver.
    # Exclude commands, empty/local turns and all BLOCKED turns, even if some
    # preceding provider requests within those turns succeeded.
    judged_llm = [r for r in records if r.get("llm_calls", 0) > 0 and r["status"] == "UNREVIEWED"
                  and r.get("input", "").strip() and not r["input"].startswith("/")]
    cli_preflight_and_question = []
    for line in (directory / "cli_run.log").read_text(encoding="utf-8").splitlines():
        if "intent raw response" in line or "agent raw response" in line or "agent repaired response" in line:
            cli_preflight_and_question.append(line)
    summary = {"updated_utc": datetime.now(timezone.utc).isoformat(), "coverage": len(covered),
        "missing_ids": sorted({f"T{n:03}" for n in range(1, 121)} - set(covered)), "attempted_turns": len(records),
        "instrumented_llm_calls": len(logs), "instrumented_llm_call_success": sum(bool(x.get("success")) for x in logs),
        "instrumented_llm_call_failures": sum(not x.get("success") for x in logs),
        "actual_user_messages_with_judgeable_llm_response_conservative": len(judged_llm),
        "cli_raw_response_records": len(cli_preflight_and_question),
        "provider_diagnostic_requests": 1 if (directory / "provider_diagnostic.json").exists() else 0,
        "all_actual_llm_requests": len(logs) + len(cli_preflight_and_question) + (1 if (directory / "provider_diagnostic.json").exists() else 0),
        "local_turns_without_llm": sum(r.get("llm_calls", 0) == 0 for r in records),
        "original_blocked_turns": sum(r["status"] == "BLOCKED" for r in records),
        "real_aspen_actions": len(executions), "evaluations_used": len(evaluation_rows),
        "evaluations_remaining": 64 - len(evaluation_rows),
        "conservative_action_wall_seconds": sum(e["action_wall_seconds"] for e in executions.values()),
        "all_source_files_unchanged": all(x["unchanged"] for x in hashes),
        "status": "部分完成，待逐项复核及未解决限制处理"}
    write(directory / "summary.json", summary)
    # Test transcripts are intentionally preserved; check the configured secret
    # without ever printing or writing the secret itself.
    from src.agents.lead.llm_client import load_llm_config
    cfg = load_llm_config()
    key = getattr(cfg, "api_key", None)
    hits = []
    if key:
        for path in directory.rglob("*"):
            if path.is_file() and path.suffix.lower() in {".json", ".jsonl", ".log", ".txt", ".md", ".csv", ".yaml"}:
                if key.encode() in path.read_bytes():
                    hits.append(str(path.relative_to(directory)))
    write(directory / "secret_scan.json", {"credential_configured": bool(key), "literal_secret_hit_files": hits})
    scripts = directory / "replay"
    scripts.mkdir(exist_ok=True)
    for name in ("agent_acceptance.py", "agent_acceptance_cli.py", "agent_acceptance_export.py", "agent_acceptance_review.py"):
        shutil.copy2(ROOT / "tools" / name, scripts / name)
    for path in (ROOT / ".tmp").glob("acceptance_*.json"):
        shutil.copy2(path, scripts / path.name)
    shutil.copy2(ROOT / "tests/test_acceptance_regressions.py", scripts / "acceptance_regressions.py.txt")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
