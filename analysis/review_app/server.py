"""Review server for the Cartwheel error-analysis interface.

Reuses the API contract from analysis/server.py (samples, annotations,
patterns, suggestions, graph) with state files under analysis/state/.
Serves the custom review UI from this directory.

    python analysis/review_app/server.py                # serve on :8020
    python analysis/review_app/server.py --port 8021
"""

from __future__ import annotations

import argparse
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
STATE_DIR = HERE.parent / "state"

API_FILES: dict[str, Path] = {
    "/api/samples": STATE_DIR / "samples.json",
    "/api/annotations": STATE_DIR / "annotations.json",
    "/api/graph": STATE_DIR / "graph.json",
    "/api/patterns": STATE_DIR / "patterns.json",
    "/api/suggestions": STATE_DIR / "suggestions.json",
    "/api/manifest": STATE_DIR / "sample_manifest.json",
}

JUDGES_DIR = STATE_DIR / "judges"
HW5_LABELS_DIR = STATE_DIR / "hw5_labels"
SPLITS_FILE = STATE_DIR / "splits.json"

API_DEFAULTS: dict[str, Any] = {
    "/api/samples": [],
    "/api/annotations": [],
    "/api/graph": {"nodes": [], "clusters": []},
    "/api/patterns": {},
    "/api/suggestions": [],
    "/api/manifest": {"batches": [], "reviewed": [], "total_reviewed": 0},
}


def _read_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text())
    except (json.JSONDecodeError, OSError):
        return default


def _write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2))
    tmp.replace(path)


class ReviewHandler(BaseHTTPRequestHandler):

    def log_message(self, fmt: str, *args: Any) -> None:
        return

    def _send_json(self, data: Any, status: int = 200) -> None:
        body = json.dumps(data).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()
        self.wfile.write(body)

    def _send_file(self, path: Path, content_type: str) -> None:
        if not path.exists():
            self._send_json({"error": f"not found: {path.name}"}, status=404)
            return
        body = path.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _read_body(self) -> Any:
        length = int(self.headers.get("Content-Length", 0))
        if length == 0:
            return None
        raw = self.rfile.read(length)
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            return None

    def do_OPTIONS(self) -> None:
        self._send_json({}, status=204)

    def do_GET(self) -> None:
        path = self.path.split("?", 1)[0]
        if path in ("/", "/index.html"):
            self._send_file(HERE / "index.html", "text/html; charset=utf-8")
            return
        if path.startswith("/static/"):
            asset = HERE / path[len("/static/"):]
            if asset.is_file() and HERE in asset.resolve().parents:
                self._send_file(asset, _guess_type(asset))
                return
        if path in API_FILES:
            data = _read_json(API_FILES[path], API_DEFAULTS[path])
            self._send_json(data)
            return
        if path == "/api/labels":
            self._send_json(_read_all_labels())
            return
        if path.startswith("/api/hw5-judge"):
            self._send_json(_build_hw5_judge_data())
            return
        self._send_json({"error": f"unknown path: {path}"}, status=404)

    def do_POST(self) -> None:
        path = self.path.split("?", 1)[0]
        if path not in API_FILES:
            self._send_json({"error": f"cannot POST to {path}"}, status=404)
            return
        data = self._read_body()
        if data is None:
            self._send_json({"error": "expected a JSON body"}, status=400)
            return
        synced = 0
        if path == "/api/annotations":
            try:
                synced = _sync_annotation_scores(data)
            except Exception as exc:
                _write_json(API_FILES[path], data)
                self._send_json(
                    {"error": f"Langfuse score write failed: {exc}", "cached_locally": True},
                    status=502,
                )
                return
        _write_json(API_FILES[path], data)
        result: dict[str, Any] = {"ok": True, "count": _count(data)}
        if synced:
            result["langfuse_scores_written"] = synced
        self._send_json(result)


def _read_all_labels() -> list[dict[str, Any]]:
    labels_dir = STATE_DIR / "labels"
    if not labels_dir.is_dir():
        return []
    result = []
    for f in sorted(labels_dir.glob("*.jsonl")):
        mode = f.stem
        for line in f.read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
                row["mode"] = mode
                result.append(row)
            except json.JSONDecodeError:
                continue
    return result


def _build_hw5_judge_data() -> dict[str, Any]:
    """Load judge predictions, critiques, and human labels for the HW5 review overlay."""
    all_splits = _read_json(SPLITS_FILE, {})

    history = _read_json(JUDGES_DIR / "_history_tool_result_incomplete.json", {"versions": []})
    versions = history.get("versions", [])
    if not versions:
        return {"ok": False, "error": "no judge versions found"}

    latest_id = versions[-1]["judge_id"]
    judge_file = JUDGES_DIR / f"{latest_id}.json"
    judge = _read_json(judge_file, {})
    if not judge:
        return {"ok": False, "error": f"judge file not found: {latest_id}"}

    mode = judge.get("mode", "tool_result_incomplete")
    splits = all_splits.get(mode, {})
    dev_ids = set(splits.get("dev", []))
    test_ids = set(splits.get("test", []))
    train_ids = set(splits.get("train", []))

    labels: dict[str, int] = {}
    for f in sorted(HW5_LABELS_DIR.glob("*.jsonl")) if HW5_LABELS_DIR.is_dir() else []:
        for line in f.read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
                labels[row["trace_id"]] = row["label"]
            except (json.JSONDecodeError, KeyError):
                continue

    prompt_hash = judge.get("prompt_hash", "")
    preds = judge.get("predictions", {}).get(prompt_hash, {})
    critiques = judge.get("critiques", {}).get(prompt_hash, {})

    entries: dict[str, Any] = {}
    disagreements: list[str] = []
    for tid in sorted(dev_ids):
        human = labels.get(tid)
        verdict = preds.get(tid)
        if human is None or verdict is None:
            continue
        is_disagree = human != verdict
        entry = {
            "human_label": human,
            "judge_verdict": verdict,
            "critique": critiques.get(tid, ""),
            "split": "dev",
            "is_disagreement": is_disagree,
        }
        entries[tid] = entry
        if is_disagree:
            disagreements.append(tid)

    for tid in sorted(train_ids):
        human = labels.get(tid)
        if human is None:
            continue
        entries[tid] = {
            "human_label": human,
            "judge_verdict": preds.get(tid),
            "critique": critiques.get(tid, ""),
            "split": "train",
            "is_disagreement": False,
        }

    return {
        "ok": True,
        "judge_id": latest_id,
        "mode": judge.get("mode", ""),
        "disagreement_ids": disagreements,
        "entries": entries,
    }


def _count(data: Any) -> int:
    if isinstance(data, (list, dict)):
        return len(data)
    return 0


def _annotation_list(data: Any) -> list[dict[str, Any]]:
    if isinstance(data, dict):
        data = data.get("annotations", [])
    return [a for a in data if isinstance(a, dict)] if isinstance(data, list) else []


def _sync_annotation_scores(data: Any) -> int:
    try:
        from analysis.helpers import langfuse_io
    except Exception:
        return 0
    if not langfuse_io.is_configured():
        return 0
    written = 0
    client = langfuse_io._client()
    for ann in _annotation_list(data):
        trace_id = ann.get("trace_id")
        mode = ann.get("mode")
        label = ann.get("label")
        if not trace_id or not mode or label not in (0, 1, "0", "1"):
            continue
        langfuse_io.write_label_score(
            trace_id=str(trace_id), mode=str(mode),
            label=int(label), comment=ann.get("note"), client=client,
        )
        written += 1
    return written


def _guess_type(path: Path) -> str:
    return {
        ".html": "text/html; charset=utf-8",
        ".css": "text/css",
        ".js": "text/javascript",
        ".json": "application/json",
        ".svg": "image/svg+xml",
    }.get(path.suffix, "application/octet-stream")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8020)
    parser.add_argument("--host", default="127.0.0.1")
    args = parser.parse_args()
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    server = ThreadingHTTPServer((args.host, args.port), ReviewHandler)
    url = f"http://{args.host}:{args.port}/"
    print(f"review interface on {url}")
    print(f"serving state from {STATE_DIR}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nshutting down")
        server.shutdown()


if __name__ == "__main__":
    main()
