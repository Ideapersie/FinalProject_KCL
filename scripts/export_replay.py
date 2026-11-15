r"""
scripts/export_replay.py: build the static replay demo from the logged runs.

The live demo (scripts/run_demo.py) needs a 1.9 GB GGUF and 2.4 GB of indexes.
This script needs neither. It joins three logged policies on the same 200
MIRAGE-MMLU questions:

    P3 closed-book       results/raw_logs/p3_mirage200.jsonl
    P1 always-retrieve   results/raw_logs/p1_medcorp_mcq_chunks.jsonl  (has retrieved chunks)
    P5 gated (ensemble)  results/raw_logs/p5_medcorp_mcq.jsonl         (has per-token entropy, votes)

with the question text from data/raw/mirage/benchmark.json, and writes one
self-contained HTML page (data inlined) that replays exactly what the gate saw
and decided. Nothing is re-run and nothing is invented: every number on the page
comes from these logs, and the headline counts are recomputed here rather than
copied from the report.

Usage:
    python scripts/export_replay.py
    python scripts/export_replay.py --out docs/demo/index.html --chunk-chars 400
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))  # use the project's own scorer, not a copy
LOGS = ROOT / "results" / "raw_logs"

# Scaling arm: (label, closed-book log, always-retrieve log, gated log).
SCALING_RUNS = [
    ("Llama-3.2-3B", "p3_mirage200.jsonl", "p1_medcorp_mcq.jsonl", "p5_medcorp_mcq.jsonl"),
    ("Qwen2.5-7B", "p3_qwen7b_mcq.jsonl", "p1_qwen7b_mcq.jsonl", "p5_qwen7b_mcq.jsonl"),
    ("Qwen2.5-14B", "p3_qwen14b_mcq.jsonl", "p1_qwen14b_mcq.jsonl", "p5_qwen14b_mcq.jsonl"),
]

PLACEHOLDER = "/*__REPLAY_DATA__*/null"


def read_jsonl(path: Path) -> Dict[str, Dict[str, Any]]:
    """Load a run log keyed by question_id."""
    out: Dict[str, Dict[str, Any]] = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rec = json.loads(line)
                out[rec["question_id"]] = rec
    return out


def index_benchmark(path: Path) -> Dict[str, Dict[str, Any]]:
    """MIRAGE benchmark.json is subset-keyed; flatten to {qid: record}."""
    data = json.loads(path.read_text(encoding="utf-8"))
    flat: Dict[str, Dict[str, Any]] = {}
    for subset, recs in data.items():
        if isinstance(recs, dict):
            for qid, rec in recs.items():
                flat[qid] = rec
    return flat


def category(closed_ok: bool, retrieve_ok: bool) -> str:
    """How retrieval changed the closed-book answer."""
    if closed_ok and not retrieve_ok:
        return "broke"
    if not closed_ok and retrieve_ok:
        return "fixed"
    return "both_right" if closed_ok else "both_wrong"


def flip_counts(closed: Dict[str, Any], retrieve: Dict[str, Any], gated: Dict[str, Any]) -> Dict[str, Any]:
    ids = sorted(closed.keys() & retrieve.keys() & gated.keys())
    n = len(ids)
    broke = sum(category(closed[i]["is_correct"], retrieve[i]["is_correct"]) == "broke" for i in ids)
    fixed = sum(category(closed[i]["is_correct"], retrieve[i]["is_correct"]) == "fixed" for i in ids)

    def acc(d: Dict[str, Any]) -> float:
        return round(sum(bool(d[i]["is_correct"]) for i in ids) / n, 4) if n else 0.0

    return {
        "n": n,
        "broke": broke,
        "fixed": fixed,
        "acc_closed": acc(closed),
        "acc_retrieve": acc(retrieve),
        "acc_gated": acc(gated),
        "gated_retrieval_rate": round(sum(bool(gated[i]["retrieval_triggered"]) for i in ids) / n, 4) if n else 0.0,
    }


def letter(answer_text: Optional[str]) -> Optional[str]:
    """Answer letter as the evaluation scored it (None if the model chose no option)."""
    from medrag_adaptive.evaluation.scoring import extract_letter
    return extract_letter(answer_text) if answer_text else None


def excerpt(answer_text: Optional[str], n: int = 240) -> str:
    t = " ".join((answer_text or "").split())
    return t[:n] + ("..." if len(t) > n else "")


def answer_view(rec: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "letter": letter(rec.get("answer_text")),
        "correct": bool(rec["is_correct"]),
        "text": excerpt(rec.get("answer_text")),
    }


def gate_view(rec: Dict[str, Any]) -> Dict[str, Any]:
    details = (rec.get("qvault") or {}).get("gate_details") or {}
    members = details.get("members") or {}
    ent = members.get("entropy") or {}
    mar = members.get("margin") or {}
    probe = members.get("hallucination_probe") or {}
    return {
        "decision": rec.get("gate_decision"),
        "votes": details.get("votes") or {},
        "retrieve_votes": details.get("retrieve_votes"),
        "min_votes": details.get("min_votes"),
        "entropy": {
            "mean": ent.get("mean_entropy"),
            "threshold": ent.get("threshold"),
            "per_token": [round(x, 4) for x in (ent.get("per_token_entropy") or [])],
        },
        "margin": {"mean": mar.get("mean_margin"), "threshold": mar.get("threshold")},
        "probe": {
            "agreement": probe.get("agreement"),
            "letter_a": probe.get("letter_a"),
            "letter_b": probe.get("letter_b"),
        },
    }


def chunks_view(rec: Dict[str, Any], chunk_chars: int, top_k: int) -> List[Dict[str, Any]]:
    chunks = ((rec.get("qvault") or {}).get("retrieved_chunks") or [])[:top_k]
    out = []
    for c in chunks:
        text = c.get("text") or ""
        out.append({
            "source": c.get("source"),
            "title": c.get("title"),
            "score": round(float(c.get("score") or 0.0), 4),
            "text": text[:chunk_chars] + ("..." if len(text) > chunk_chars else ""),
        })
    return out


def build_records(
    closed: Dict[str, Any],
    retrieve: Dict[str, Any],
    gated: Dict[str, Any],
    bench: Dict[str, Any],
    chunk_chars: int = 400,
    top_k: int = 3,
) -> List[Dict[str, Any]]:
    records = []
    for qid in sorted(closed.keys() & retrieve.keys() & gated.keys()):
        q = bench.get(qid)
        if q is None:
            continue
        c, r, g = closed[qid], retrieve[qid], gated[qid]
        records.append({
            "id": qid,
            "topic": qid.rsplit("-", 1)[0].replace("_", " "),
            "question": q["question"],
            "options": q.get("options") or {},
            "gold": str(q["answer"]),
            "category": category(bool(c["is_correct"]), bool(r["is_correct"])),
            "closed": answer_view(c),
            "retrieve": answer_view(r),
            "gated": {**answer_view(g), "retrieved": bool(g.get("retrieval_triggered"))},
            "gate": gate_view(g),
            "chunks": chunks_view(r, chunk_chars, top_k),
        })
    return records


def build_payload(logs: Path, dataset: Path, chunk_chars: int, top_k: int) -> Dict[str, Any]:
    bench = index_benchmark(dataset)
    closed = read_jsonl(logs / "p3_mirage200.jsonl")
    retrieve = read_jsonl(logs / "p1_medcorp_mcq_chunks.jsonl")
    gated = read_jsonl(logs / "p5_medcorp_mcq.jsonl")

    scaling = []
    for label, p3, p1, p5 in SCALING_RUNS:
        paths = [logs / p3, logs / p1, logs / p5]
        if all(p.exists() for p in paths):
            scaling.append({"model": label, **flip_counts(*(read_jsonl(p) for p in paths))})

    return {
        "model": "Llama-3.2-3B-Instruct Q4_K_M (CPU)",
        "corpus": "MedRAG textbooks + 300K PubMed abstracts, 425,847 chunks, BM25 + dense via RRF",
        "summary": flip_counts(closed, retrieve, gated),
        "scaling": scaling,
        "questions": build_records(closed, retrieve, gated, bench, chunk_chars, top_k),
    }


def render(template: str, payload: Dict[str, Any]) -> str:
    if PLACEHOLDER not in template:
        raise ValueError(f"template is missing the {PLACEHOLDER!r} placeholder")
    # Escape "</" so question text can never close the <script> tag early.
    blob = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).replace("</", "<\\/")
    return template.replace(PLACEHOLDER, blob)


def main(argv: Optional[Iterable[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--logs", type=Path, default=LOGS)
    ap.add_argument("--dataset", type=Path, default=ROOT / "data" / "raw" / "mirage" / "benchmark.json")
    ap.add_argument("--template", type=Path, default=ROOT / "docs" / "demo" / "template.html")
    ap.add_argument("--out", type=Path, default=ROOT / "docs" / "demo" / "index.html")
    ap.add_argument("--json-out", type=Path, default=None, help="also write the payload as JSON")
    ap.add_argument("--chunk-chars", type=int, default=400)
    ap.add_argument("--top-k", type=int, default=3)
    args = ap.parse_args(list(argv) if argv is not None else None)

    payload = build_payload(args.logs, args.dataset, args.chunk_chars, args.top_k)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(render(args.template.read_text(encoding="utf-8"), payload), encoding="utf-8")
    if args.json_out:
        args.json_out.write_text(json.dumps(payload, indent=1, ensure_ascii=False), encoding="utf-8")

    s = payload["summary"]
    print(f"{len(payload['questions'])} questions -> {args.out}")
    print(f"retrieval broke {s['broke']}, fixed {s['fixed']}; "
          f"closed {s['acc_closed']:.1%}, always-retrieve {s['acc_retrieve']:.1%}, gated {s['acc_gated']:.1%}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
