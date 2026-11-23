"""
Call POST /api/chat for each baked-in sample question and write a Markdown report.

Run from repo root with the API up, e.g.:

    cd backend && uvicorn src.app:app --reload --port 9481

    python -m experiments.scripts.05_surprise_me_answers.main
"""

from __future__ import annotations

import argparse
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import requests

from questions import QUESTIONS

SNIPPET_MAX_LEN = 240


def health_check(base_url: str, timeout: float) -> bool:
    url = f"{base_url.rstrip('/')}/api/health"
    try:
        resp = requests.get(url, timeout=timeout)
        resp.raise_for_status()
        data = resp.json()
        docs = data.get("documents_loaded", "?")
        print(f"[OK] API healthy — {docs} documents loaded")
        return True
    except Exception as e:
        print(f"[ERROR] API health check failed: {e}")
        return False


def call_chat(question: str, base_url: str, timeout: float) -> tuple[str | None, list[dict], float | None, str | None]:
    url = f"{base_url.rstrip('/')}/api/chat"
    payload = {"query": question, "history": []}
    t0 = time.perf_counter()
    try:
        resp = requests.post(url, json=payload, timeout=timeout)
        latency = time.perf_counter() - t0
        resp.raise_for_status()
        data = resp.json()
        answer = data.get("content") or ""
        sources = data.get("sources") or []
        if not isinstance(sources, list):
            sources = []
        return answer, sources, round(latency, 2), None
    except Exception as e:
        return None, [], None, str(e)


def _fence_delimiters(body: str) -> tuple[str, str]:
    n = 3
    while "`" * n in body:
        n += 1
    fence = "`" * n
    return fence, fence


def _truncate_snippet(text: str) -> str:
    t = text.replace("\n", " ").strip()
    if len(t) <= SNIPPET_MAX_LEN:
        return t
    return t[: SNIPPET_MAX_LEN - 3] + "..."


def build_markdown(
    base_url: str,
    results: list[tuple[str, str | None, list[dict], float | None, str | None]],
) -> str:
    lines: list[str] = [
        "# Surprise-me questions — API snapshot",
        "",
        f"- **Generated:** {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}",
        f"- **Base URL:** {base_url}",
        f"- **Endpoint:** `/api/chat`",
        "",
    ]
    for i, (question, answer, sources, latency, err) in enumerate(results, start=1):
        lines.append(f"## {i}. {question}")
        lines.append("")
        if err is not None:
            lines.append(f"**Error:** {err}")
            lines.append("")
            lines.append("---")
            lines.append("")
            continue
        open_f, close_f = _fence_delimiters(answer or "")
        lines.append("**Answer:**")
        lines.append("")
        lines.append(open_f)
        lines.append(answer or "")
        lines.append(close_f)
        lines.append("")
        lines.append("**Sources:**")
        if not sources:
            lines.append("- _(none)_")
        else:
            for s in sources:
                if not isinstance(s, dict):
                    lines.append(f"- `{s}`")
                    continue
                doc_id = s.get("docId", "")
                snippet = s.get("snippet", "")
                lines.append(f"- **`{doc_id}`** — {_truncate_snippet(str(snippet))}")
        lines.append("")
        lines.append(f"**Latency:** {latency}s")
        lines.append("")
        lines.append("---")
        lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Snapshot /api/chat answers for sample questions → Markdown")
    parser.add_argument("--base-url", default="http://localhost:9481", help="API base URL")
    parser.add_argument("--timeout", type=float, default=120.0, help="HTTP timeout per request (seconds)")
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output .md path (default: output/surprise_me_<timestamp>.md under script dir)",
    )
    parser.add_argument("--delay", type=float, default=0.0, help="Seconds to sleep between requests")
    args = parser.parse_args()

    base_url = args.base_url.rstrip("/")
    if not health_check(base_url, min(10.0, args.timeout)):
        sys.exit(1)

    out_path = args.output
    if out_path is None:
        ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        out_dir = SCRIPT_DIR / "output"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"surprise_me_{ts}.md"
    else:
        out_path = out_path.resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)

    results: list[tuple[str, str | None, list[dict], float | None, str | None]] = []
    total = len(QUESTIONS)
    for idx, question in enumerate(QUESTIONS, start=1):
        print(f"[{idx}/{total}] {question[:60]}{'…' if len(question) > 60 else ''}")
        answer, sources, latency, err = call_chat(question, base_url, args.timeout)
        results.append((question, answer, sources, latency, err))
        if args.delay > 0 and idx < total:
            time.sleep(args.delay)

    md = build_markdown(base_url, results)
    out_path.write_text(md, encoding="utf-8")
    print(f"[OK] Wrote {out_path}")


if __name__ == "__main__":
    main()
