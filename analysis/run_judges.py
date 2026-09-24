"""HW5: LLM judge pipeline for tool_result_incomplete."""

from __future__ import annotations

import json
import os
from pathlib import Path

import dotenv

dotenv.load_dotenv()

os.environ.setdefault("CARTWHEEL_JUDGE_TRACE_SOURCE", str(Path("analysis/state/hw5_trace_inputs.json")))

MODE = "tool_result_incomplete"
STATE_DIR = Path("analysis/state")
PROMPTS_DIR = Path("analysis/prompts")
REPORT_DIR = Path("analysis/report")
TRACE_SOURCE = Path("traces/support_traces.json")
SCENARIO_SOURCES = [
    Path("scenarios/hw5-tri-results.jsonl"),
    Path("scenarios/hw5-tri-batch2-results.jsonl"),
]
HW5_LABELS = STATE_DIR / "hw5_labels" / f"{MODE}.jsonl"
TRACE_INPUTS = STATE_DIR / "hw5_trace_inputs.json"


def prepare_inputs() -> list[dict]:
    """Extract labeled traces into judge-ready inputs.

    Reads support_traces.json and HW5 scenario results, normalizes each
    trace, and keeps only those with HW5 labels. Saves trace_id + message
    list — no labels, annotations, or scenario metadata.
    """
    from analysis.helpers.normalization import normalize_trace

    with open(TRACE_SOURCE) as f:
        data = json.load(f)
    raw_traces = data["traces"]

    labels = [json.loads(line) for line in HW5_LABELS.open()]
    labeled_ids = {l["trace_id"] for l in labels}

    records = []
    for raw in raw_traces:
        tid = raw.get("id", raw.get("trace_id"))
        if tid not in labeled_ids:
            continue
        norm = normalize_trace(raw)
        records.append({"trace_id": norm["trace_id"], "trace": norm["trace"]})

    for src in SCENARIO_SOURCES:
        if not src.exists():
            continue
        for line in src.open():
            scenario = json.loads(line)
            sid = scenario.get("scenario_id", "")
            if sid not in labeled_ids or not scenario.get("turns"):
                continue
            raw_record = {"id": sid, "turns": scenario["turns"]}
            norm = normalize_trace(raw_record)
            records.append({"trace_id": norm["trace_id"], "trace": norm["trace"]})

    found_ids = {r["trace_id"] for r in records}
    missing = labeled_ids - found_ids
    if missing:
        print(f"Warning: {len(missing)} labeled traces not in source: {sorted(missing)[:5]}")

    TRACE_INPUTS.write_text(json.dumps(records, indent=2, default=str))
    print(f"Saved {len(records)} trace inputs to {TRACE_INPUTS}")
    return records


def split_data(mode: str = MODE) -> dict[str, list[str]]:
    """Split HW5 labels into train/dev/test (20/40/40)."""
    from analysis.helpers import split_labels

    records = json.loads(TRACE_INPUTS.read_text())
    eligible_ids = [r["trace_id"] for r in records]

    splits = split_labels(
        mode,
        fractions=(0.20, 0.40, 0.40),
        seed=7,
        min_per_class=10,
        eligible_trace_ids=eligible_ids,
    )

    labels_by_id = {}
    for line in HW5_LABELS.open():
        rec = json.loads(line)
        labels_by_id[rec["trace_id"]] = rec["label"]

    for name in ("train", "dev", "test"):
        ids = splits[name]
        n_pass = sum(1 for tid in ids if labels_by_id.get(tid) == 1)
        n_fail = sum(1 for tid in ids if labels_by_id.get(tid) == 0)
        print(f"  {name:5s}: {len(ids):3d} total  ({n_pass} Pass, {n_fail} Fail)")

    return splits


def run_development(mode: str, prompt_path: str, judge_model: str = "gpt-4o-mini") -> dict:
    """Register a prompt version and run it on the dev split."""
    from analysis.helpers import register_judge, run_judge, judge_alignment

    prompt_text = Path(prompt_path).read_text()
    record = register_judge(mode=mode, prompt_text=prompt_text, judge_model=judge_model)
    judge_id = record["judge_id"]
    print(f"Registered {judge_id} (hash: {record['prompt_hash']})")

    run_judge(judge_id, split="dev", batch_size=10)
    result = judge_alignment(judge_id, split="dev")

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    report_path = REPORT_DIR / f"dev-{judge_id}.json"
    report_path.write_text(json.dumps(result, indent=2))
    print(f"Dev results saved to {report_path}")
    return result


def run_test(judge_id: str) -> dict:
    """Freeze the judge and evaluate on held-out test traces."""
    from analysis.helpers import freeze_judge, run_judge, judge_alignment

    freeze_judge(judge_id)
    print(f"Frozen {judge_id}")

    run_judge(judge_id, split="test", batch_size=10)
    result = judge_alignment(judge_id, split="test")

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    report_path = REPORT_DIR / f"test-{judge_id}.json"
    report_path.write_text(json.dumps(result, indent=2))
    print(f"Test results saved to {report_path}")
    return result


if __name__ == "__main__":
    import sys

    cmd = sys.argv[1] if len(sys.argv) > 1 else "prepare"
    if cmd == "prepare":
        prepare_inputs()
    elif cmd == "split":
        split_data()
    elif cmd == "dev":
        prompt = sys.argv[2] if len(sys.argv) > 2 else f"analysis/prompts/{MODE}-v0.txt"
        run_development(MODE, prompt)
    elif cmd == "test":
        jid = sys.argv[2]
        run_test(jid)
    else:
        print(f"Unknown command: {cmd}")
