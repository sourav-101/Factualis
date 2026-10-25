from __future__ import annotations

import argparse
import html
import json
import re
import time
from pathlib import Path

import uvicorn
from fastapi import FastAPI
from fastapi.responses import FileResponse, JSONResponse
from openai import OpenAI
from pydantic import BaseModel

from factualis.policy_library import Retriever, load_corpus
from factualis.prompts import format_context
from web.answering import EXAMPLES, STATUS, ask, doc_name, precompute_examples, verify

STATIC = Path(__file__).parent / "static"


class AskRequest(BaseModel):
    question: str
    verified_mode: bool = False
    constrained: bool = True


def mark(text: str, quotes: list[str]) -> str:
    out = html.escape(text)
    for q in quotes:
        core = re.sub(r"\s+", "", html.escape(q))
        if len(core) < 8:
            continue
        pattern = r"\s*".join(re.escape(ch) for ch in core)
        out = re.sub(f"(?i)({pattern})", r"<mark>\1</mark>", out, count=1)
    return out


def clean_clause(clause: str) -> str:
    return re.sub(r"\s*(\(cont\. \d+\)|\[\d+\])", "", clause or "").strip() or clause


def build_answer(raw: str | None, context: str, chunks: list[dict]) -> dict | None:
    if raw is None:
        return None
    out, quotes, values = verify(raw, context)
    if out is None:
        return {"valid": False, "raw": raw[:1500]}
    verdict, label, tone = STATUS.get(out.epistemic_status, (out.epistemic_status, "", "none"))
    ids = {(c["doc_id"], c["clause"]): i for i, c in enumerate(chunks)}

    def source_index(cite) -> int | None:
        for i, c in enumerate(chunks):
            if re.sub(r"\s+", "", cite.exact_quote.lower()) in re.sub(r"\s+", "", c["text"].lower()):
                return i
        return ids.get((cite.doc_id, cite.clause_or_section))

    return {
        "valid": True,
        "status": out.epistemic_status,
        "status_label": label,
        "tone": tone,
        "verdict": verdict,
        "summary": out.summary_answer,
        "requirements": [{"value": r.exact_value, "name": r.parameter_name.replace("_", " "), "condition": r.trigger_condition,
                          "verified": ok} for r, ok in values],
        "citations": [{"doc": doc_name(c.doc_id), "clause": clean_clause(c.clause_or_section), "quote": c.exact_quote, "verified": ok,
                       "source": source_index(c)} for c, ok in quotes],
        "gaps": list(out.unstated_regulatory_gaps),
        "checks_passed": sum(ok for _, ok in quotes) + sum(ok for _, ok in values),
        "checks_total": len(quotes) + len(values),
    }


def create_app(aligned_url: str, base_url: str | None, base_cache: Path | None, corpus: Path) -> FastAPI:
    retriever = Retriever(load_corpus(corpus))
    aligned = OpenAI(base_url=aligned_url, api_key="EMPTY", timeout=90, max_retries=1)
    base = OpenAI(base_url=base_url, api_key="EMPTY", timeout=90, max_retries=1) if base_url else None
    cache = json.loads(base_cache.read_text(encoding="utf-8")) if base_cache and base_cache.is_file() else {}
    app = FastAPI(title="Factualis")

    @app.get("/")
    def index():
        return FileResponse(STATIC / "index.html")

    @app.get("/api/examples")
    def examples():
        return [{"label": label, "question": q} for label, q in EXAMPLES]

    @app.post("/api/ask")
    def ask_endpoint(req: AskRequest):
        question = req.question.strip()
        if not question:
            return JSONResponse({"error": "Please type a question."}, status_code=400)
        chunks = retriever.search(question, 6)
        context = format_context(chunks)
        t = time.time()
        try:
            a_raw = ask(aligned, question, chunks, req.constrained)
        except Exception as e:
            print(f"model call failed: {e!r}", flush=True)
            return JSONResponse({"error": "The answering model is not available right now. Please try again in a minute."}, status_code=503)
        seconds = time.time() - t
        b_raw = ask(base, question, chunks, req.constrained) if base else cache.get(question, {}).get(str(req.constrained))
        answer, before = build_answer(a_raw, context, chunks), build_answer(b_raw, context, chunks)
        quotes = [c["quote"] for c in (answer or {}).get("citations", []) if c["verified"]]
        return {
            "question": question,
            "seconds": round(seconds, 1),
            "answer": answer,
            "before_training": before,
            "held": bool(req.verified_mode and answer and answer.get("valid") and answer["checks_passed"] < answer["checks_total"]),
            "sources": [{"doc": doc_name(c["doc_id"]), "clause": clean_clause(c["clause"]), "html": mark(c["text"], quotes)} for c in chunks],
        }

    return app


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--aligned-port", default="30000")
    ap.add_argument("--base-port")
    ap.add_argument("--base-cache", type=Path, default=Path("output/demo_base_cache.json"))
    ap.add_argument("--corpus", type=Path, default=Path("data/policy_clauses.jsonl"))
    ap.add_argument("--no-compare", action="store_true", help="hide the 'Before training' comparison")
    ap.add_argument("--precompute-examples", type=Path, help="save the served model's answers to the example questions to this file and exit")
    a = ap.parse_args()
    if a.precompute_examples:
        precompute_examples(a.aligned_port, a.precompute_examples, Retriever(load_corpus(a.corpus)))
        return
    app = create_app(f"http://localhost:{a.aligned_port}/v1", f"http://localhost:{a.base_port}/v1" if a.base_port else None,
                     None if a.no_compare else a.base_cache, a.corpus)
    uvicorn.run(app, host="0.0.0.0", port=a.port, log_level="warning")


if __name__ == "__main__":
    main()
