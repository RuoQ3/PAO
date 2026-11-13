"""Run the production REPL in a separate process, preserving its exact IO."""
from pathlib import Path
import argparse
import json
import subprocess
import sys
import time


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--directory", required=True)
    args = parser.parse_args()
    directory = Path(args.directory).resolve()
    prompt = "读取当前YAML，列出三个设计变量的完整Aspen路径及单位。"
    commands = [prompt, "/details", "/value p1", "/value p999",
                f"/export {directory / 'cli_answer.json'}",
                f"/export {directory / 'cli_answer.md'}",
                f"/export {directory / 'cli_answer.json'}", "/capabilities", "exit"]
    stdin = "\n".join(commands) + "\n"
    (directory / "cli_stdin.txt").write_text(stdin, encoding="utf-8")
    command = [sys.executable, "-m", "src.main", str(directory / "ga.yaml"),
               "--conversation", "--db", str(directory / "simulation.db"),
               "--node-db", "cases/demo_case/output/node.db",
               "--log-file", str(directory / "cli_run.log")]
    started = time.monotonic()
    completed = subprocess.run(command, input=stdin, encoding="utf-8", errors="replace",
                               capture_output=True, timeout=300)
    (directory / "cli_stdout.txt").write_text(completed.stdout, encoding="utf-8")
    (directory / "cli_stderr.txt").write_text(completed.stderr, encoding="utf-8")
    chunks = completed.stdout.split("PAO> ")
    records = []
    for n, index in ((81, 8), (82, 2), (83, 3), (84, 4), (85, 5), (86, 6), (87, 7)):
        record = {"id": f"T{n:03}", "label": "real_cli", "input": commands[index - 1],
                  "response": chunks[index] if index < len(chunks) else completed.stdout,
                  "status": "UNREVIEWED", "llm_calls": 0,
                  "elapsed_seconds": time.monotonic() - started,
                  "preconditions": {"command": command, "preceding_inputs": commands[:index - 1]},
                  "expected": "见 original_cases.json", "evidence": str(directory / "cli_stdout.txt"),
                  "session_id": "production-cli", "returncode": completed.returncode}
        records.append(record)
    with (directory / "cases.jsonl").open("a", encoding="utf-8") as stream:
        for record in records:
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(json.dumps({"returncode": completed.returncode, "commands": len(commands),
                      "elapsed_seconds": time.monotonic() - started}, ensure_ascii=False))


if __name__ == "__main__":
    main()
