from __future__ import annotations

import argparse
import json
import re
import shutil
from collections import Counter
from difflib import SequenceMatcher
from pathlib import Path

from factualis.policy_library import Retriever, load_corpus
from factualis.prompts import format_context
from factualis.verifier import normalize

STATUSES = {"FULLY_GROUNDED", "PARTIALLY_GROUNDED", "ABSTAIN_NOT_IN_POLICY", "REGULATORY_CONFLICT"}
EXPECTED = {"answerable", "trap", "partial"}


def export(corpus: Path, out: Path) -> None:
    chunks = sorted(load_corpus(corpus), key=lambda c: not c["doc_id"].startswith("fb_"))
    out.write_text(format_context(chunks), encoding="utf-8")
    print(f"wrote {out} ({len(chunks)} chunks, {out.stat().st_size / 1e3:.0f} KB) - attach this file to Gemini")


_STR_FIELD_RE = re.compile(r'("(?:question|exact_quote|missing|notes)": ")(.*?)("(?:, "|\}))')


def _repair(line: str) -> str:
    return _STR_FIELD_RE.sub(lambda m: m.group(1) + re.sub(r'(?<!\\)"', r'\\"', m.group(2)) + m.group(3), line)


def load_rows(path: Path) -> list[dict]:
    lines = [l.strip() for l in path.open(encoding="utf-8") if l.strip()]
    rows, fixed, bad = [], [], []
    for n, line in enumerate(lines, 1):
        line = line.removeprefix("```json").removeprefix("```").strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
            fixed.append(line)
        except json.JSONDecodeError:
            try:
                rows.append(json.loads(_repair(line)))
                fixed.append(_repair(line))
                print(f"  repaired unescaped quotes on line {n}")
            except json.JSONDecodeError as e:
                bad.append(f"line {n}: {e.msg} at col {e.colno}: ...{line[max(0, e.pos - 60):e.pos + 30]}...")
    if bad:
        print("Invalid JSON lines (fix by hand, or ask Gemini to re-emit them):")
        print("\n".join("  - " + b for b in bad))
        raise SystemExit(1)
    if fixed != lines:
        shutil.copyfile(path, path.with_suffix(path.suffix + ".bak"))
        path.write_text("\n".join(fixed) + "\n", encoding="utf-8")
        print(f"  wrote repaired file (original saved as {path.name}.bak)")
    return rows


_STOP = set("the a an of for to in is what and or be do does i we can if on under by with are it this our my".split())


def _content_tokens(q: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", q.lower())) - _STOP


def _near_duplicate(q: str, others: list[tuple[str, set[str]]]) -> str | None:
    tq = _content_tokens(q)
    for o, to in others:
        if len(tq & to) / max(1, len(tq | to)) >= 0.5 or SequenceMatcher(None, q, o).ratio() >= 0.75:
            return o
    return None


def _best_match(quote: str, chunks: list[dict]) -> tuple[float, dict]:
    nq = normalize(quote)
    return max(((SequenceMatcher(None, nq, normalize(c["text"])).find_longest_match().size / max(len(nq), 1), c) for c in chunks),
               key=lambda x: x[0])


def check(path: Path, corpus: Path, seeds: bool, against: Path | None) -> None:
    chunks = load_corpus(corpus)
    by_doc: dict[str, list[dict]] = {}
    for c in chunks:
        by_doc.setdefault(c["doc_id"], []).append(c)
    retriever = Retriever(chunks)
    rows = load_rows(path)
    other = set()
    if against and against.exists():
        other = {normalize(json.loads(l)["question"]) for l in against.open(encoding="utf-8") if l.strip()}
    other_toks = [(q, _content_tokens(q)) for q in other]

    problems, review = [], [f"# Review sheet for {path}\n"]
    seen = Counter(normalize(r.get("question", "")) for r in rows)
    for i, r in enumerate(rows, 1):
        rid = r.get("id", f"row{i}")
        q = r.get("question", "")
        label = r.get("expected") if seeds else r.get("gold_status")
        valid = EXPECTED if seeds else STATUSES
        if not q:
            problems.append(f"{rid}: missing question")
            continue
        if label not in valid:
            problems.append(f"{rid}: label {label!r} not in {sorted(valid)}")
        if seen[normalize(q)] > 1:
            problems.append(f"{rid}: duplicate question")
        if normalize(q) in other:
            problems.append(f"{rid}: also appears in {against} (train/test leak)")
        else:
            near = _near_duplicate(normalize(q), other_toks)
            if near:
                problems.append(f"{rid}: paraphrase of a question in {against} (train/test leak): {near[:80]!r}")

        review.append(f"## {rid} - {label}\n**Q:** {q}\n")
        for ev in r.get("evidence", []):
            doc, quote = ev.get("doc_id"), ev.get("exact_quote", "")
            pool = by_doc.get(doc)
            if not pool:
                problems.append(f"{rid}: unknown doc_id {doc!r}")
                continue
            hit = next((c for c in pool if normalize(quote) in normalize(c["text"])), None)
            if hit:
                if ev.get("clause") and normalize(ev["clause"]) not in normalize(hit["clause"]):
                    problems.append(f"{rid}: quote found in clause {hit['clause']!r}, not {ev['clause']!r}")
                review.append(f"- OK `{doc} | {hit['clause']}`: \"{quote}\"")
            else:
                score, near = _best_match(quote, pool)
                problems.append(f"{rid}: quote NOT verbatim in {doc} (best overlap {score:.0%} in clause {near['clause']!r}): {quote[:80]!r}")
                review.append(f"- FAIL `{doc}`: \"{quote}\"  \n  nearest clause {near['clause']}: {near['text'][:200]!r}")
        if label in ("ABSTAIN_NOT_IN_POLICY", "PARTIALLY_GROUNDED", "trap", "partial"):
            review.append("\nClosest clauses (confirm the documents are silent on the missing part):")
            for c in retriever.ranked(q, 3):
                review.append(f"- `{c['doc_id']} | {c['clause']}`: {c['text'][:220]!r}")
            if r.get("missing"):
                review.append(f"\nClaimed missing: {r['missing']}")
        else:
            if not seeds and not r.get("evidence"):
                problems.append(f"{rid}: {label} row has no evidence quotes")
            cited = {ev.get("doc_id") for ev in r.get("evidence", [])}
            others = [c for c in retriever.ranked(q, 12) if c["doc_id"] not in cited][:3]
            if others:
                review.append("\nOther documents on the same topic (could this be a conflict?):")
                for c in others:
                    review.append(f"- `{c['doc_id']} | {c['clause']}`: {c['text'][:220]!r}")
        review.append("\n- [ ] reviewed by a human\n")

    sheet = path.with_suffix(".review.md")
    sheet.write_text("\n".join(review), encoding="utf-8")
    print(f"{len(rows)} rows: {dict(Counter((r.get('expected') if seeds else r.get('gold_status')) for r in rows))}")
    print(f"docs cited: {dict(Counter(ev.get('doc_id') for r in rows for ev in r.get('evidence', [])))}")
    print(f"review sheet -> {sheet}")
    if problems:
        print(f"\n{len(problems)} problem(s):")
        for p in problems:
            print("  - " + p)
        raise SystemExit(1)
    print("all evidence quotes verified")


def merge(src: Path, dest: Path) -> None:
    have = {normalize(r["question"]) for r in load_rows(dest)} if dest.exists() else set()
    new = [r for r in load_rows(src) if normalize(r["question"]) not in have]
    with dest.open("a", encoding="utf-8") as f:
        for r in new:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"appended {len(new)} rows from {src} to {dest}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("export")
    e.add_argument("--corpus", type=Path, default=Path("data/policy_clauses.jsonl"))
    e.add_argument("--out", type=Path, default=Path("data/question_generation_prompts/policy_context.txt"))
    c = sub.add_parser("check")
    c.add_argument("path", type=Path)
    c.add_argument("--corpus", type=Path, default=Path("data/policy_clauses.jsonl"))
    c.add_argument("--seeds", action="store_true", help="validate a seeds file ({question, expected})")
    c.add_argument("--against", type=Path, help="fail if any question also appears in this file")
    mg = sub.add_parser("merge", help="append reviewed rows into the main file")
    mg.add_argument("src", type=Path)
    mg.add_argument("dest", type=Path)
    a = ap.parse_args()
    if a.cmd == "export":
        export(a.corpus, a.out)
    elif a.cmd == "merge":
        merge(a.src, a.dest)
    else:
        check(a.path, a.corpus, a.seeds, a.against)
