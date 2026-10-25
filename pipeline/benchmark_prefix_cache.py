from __future__ import annotations

import argparse
import json
import statistics
import time
import urllib.request
from pathlib import Path

from factualis.llm_client import get_client
from factualis.policy_library import Retriever, load_corpus
from factualis.prompts import build_messages
from factualis.answer_format import response_format

QUESTIONS = [
    "Who must approve an outward cross-border remittance above the limit?",
    "What review applies to a large domestic transfer by a new customer?",
    "What happens when a sanctions screening match is confirmed?",
    "How long are wire transfer records retained?",
    "What applies to transfers involving FATF increased monitoring jurisdictions?",
    "Do PEP salary credits need enhanced due diligence on every credit?",
    "Within what time must an employee raise an internal alert?",
    "Which lists are wires screened against before release?",
]


def ttft(client, messages, constrained: bool) -> float:
    t = time.time()
    kwargs = {"response_format": response_format()} if constrained else {}
    stream = client.chat.completions.create(model="default", messages=messages, temperature=0.0, max_tokens=64, stream=True, **kwargs)
    for chunk in stream:
        if chunk.choices and chunk.choices[0].delta.content:
            dt = time.time() - t
            break
    else:
        dt = time.time() - t
    for _ in stream:
        pass
    return dt


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", default="30000")
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--corpus", default="data/policy_clauses.jsonl")
    ap.add_argument("--out", default="output/eval/radix_bench.json")
    a = ap.parse_args()

    client = get_client()
    shared = Retriever(load_corpus(Path(a.corpus))).search("wire transfer approval sanctions PEP FATF retention", 6)
    qs = (QUESTIONS * 4)[: a.n]

    def flush():
        urllib.request.urlopen(urllib.request.Request(f"http://localhost:{a.port}/flush_cache", method="POST"), timeout=30).read()

    cold, warm = [], []
    for q in qs:
        flush()
        cold.append(ttft(client, build_messages(q, shared), True))
    flush()
    ttft(client, build_messages("warm-up", shared), True)
    for q in qs:
        warm.append(ttft(client, build_messages(q, shared), True))

    res = {
        "n": len(qs),
        "context_chunks": len(shared),
        "ttft_cold_ms_median": 1000 * statistics.median(cold),
        "ttft_warm_ms_median": 1000 * statistics.median(warm),
        "speedup": statistics.median(cold) / max(statistics.median(warm), 1e-6),
    }
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
