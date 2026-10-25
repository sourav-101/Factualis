import argparse
from collections import Counter
from pathlib import Path

from factualis.policy_library import build_corpus


def main() -> None:
    ap = argparse.ArgumentParser(description="Split the regulation PDFs and bank procedures into clause-level JSONL.")
    ap.add_argument("--src", nargs="+", default=["data/regulations", "data/bank_procedures"])
    ap.add_argument("--out", default="data/policy_clauses.jsonl")
    a = ap.parse_args()
    clauses = build_corpus([Path(p) for p in a.src], Path(a.out))
    for doc, n in Counter(c["doc_id"] for c in clauses).items():
        print(f"  {doc}: {n} clauses")
    print(f"wrote {len(clauses)} clauses to {a.out}")


if __name__ == "__main__":
    main()
