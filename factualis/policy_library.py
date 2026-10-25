from __future__ import annotations

import json
import re
from pathlib import Path

CLAUSE_RE = re.compile(
    r"^[ \t]*((?i:annexure|schedule)\s+[A-Z0-9]+\b|\d{1,2}(?:\.\d{1,2}){1,3}\.?|\d{1,3}\.)[ \t]+(?=[A-Za-z(\[\"'\-]|\d{1,3}[A-Z])",
    re.M,
)
INLINE_REG_RE = re.compile(r"(?<=[a-z.:\]]) (\d{1,2}[A-Z]?\. \(1\))")
MAX_CHARS = 1800
HEADING_MAX = 120
_PUNCT_FIX = str.maketrans({"�": '"', "‘": "'", "’": "'", "“": '"', "”": '"'})


def pdf_to_text(path: Path) -> str:
    from pypdf import PdfReader

    return "\n".join(page.extract_text() or "" for page in PdfReader(str(path)).pages)


def clean_text(text: str) -> str:
    keep = []
    for line in text.splitlines():
        letters = [ch for ch in line if ch.isalpha()]
        if letters and sum(ch.isascii() for ch in letters) / len(letters) < 0.7:
            continue
        if re.fullmatch(r"\s*(page\s*)?\d+(\s*of\s*\d+)?\s*", line, re.I):
            continue
        if line.strip() == "Withdrawn":
            continue
        keep.append(line)
    text = "\n".join(keep).translate(_PUNCT_FIX)
    text = INLINE_REG_RE.sub(r"\n\1", text)
    return re.sub(r"[ \t]+", " ", text)


def _windows(body: str) -> list[str]:
    parts = []
    while len(body) > MAX_CHARS:
        cut = max(body.rfind("\n", 0, MAX_CHARS), body.rfind(". ", 0, MAX_CHARS) + 1)
        if cut < MAX_CHARS // 2:
            cut = body.rfind(" ", 0, MAX_CHARS)
        parts.append(body[:cut].strip())
        body = body[cut:].strip()
    return parts + [body]


def _is_heading(body: str, m: re.Match) -> bool:
    rest = re.sub(r"^\d+", "", body[m.end():].strip())
    if len(body) > HEADING_MAX or "\n" in body or rest.lower().rstrip(".") in ("deleted", "omitted"):
        return False
    return rest.isupper() or not rest.endswith((".", ":", ";"))


def split_clauses(doc_id: str, text: str) -> list[dict]:
    text = clean_text(text)
    starts = [m.start() for m in CLAUSE_RE.finditer(text)] or [0]
    if starts[0] != 0:
        starts.insert(0, 0)
    chunks, heading, heading_num, seen = [], "", "", {}
    for i, s in enumerate(starts):
        body = text[s : starts[i + 1] if i + 1 < len(starts) else len(text)].strip()
        m = CLAUSE_RE.match(body)
        if m and _is_heading(body, m):
            heading, heading_num = body, m.group(1).strip().rstrip(".")
            continue
        if len(body) < 40:
            continue
        clause = m.group(1).strip().rstrip(".") if m else "preamble"
        seen[clause] = seen.get(clause, 0) + 1
        if seen[clause] > 1:
            clause = f"{clause} [{seen[clause]}]"
        number = clause.split(" ")[0]
        in_section = number.startswith(heading_num + ".") or heading_num.lower().startswith(("annexure", "schedule"))
        if heading and in_section and not body.startswith(heading):
            body = f"{heading}\n{body}"
        for j, part in enumerate(_windows(body)):
            cid = clause if j == 0 else f"{clause} (cont. {j})"
            chunks.append({"id": f"{doc_id}::{cid}", "doc_id": doc_id, "clause": cid, "text": part})
    return chunks


def build_corpus(src_dirs: list[Path], out_path: Path) -> list[dict]:
    chunks = []
    for d in src_dirs:
        for f in sorted(d.glob("*")):
            if f.suffix.lower() == ".pdf":
                chunks.extend(split_clauses(f.stem, pdf_to_text(f)))
            elif f.suffix.lower() in (".txt", ".md"):
                chunks.extend(split_clauses(f.stem, f.read_text(encoding="utf-8")))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for c in chunks:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")
    return chunks


def load_corpus(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]


class Retriever:
    def __init__(self, chunks: list[dict]):
        from rank_bm25 import BM25Okapi

        self.chunks = chunks
        self.bm25 = BM25Okapi([self._tok(c["doc_id"] + " " + c["text"]) for c in chunks])

    @staticmethod
    def _tok(s: str) -> list[str]:
        return re.findall(r"\w+", s.lower())

    def ranked(self, query: str, k: int = 4) -> list[dict]:
        scores = self.bm25.get_scores(self._tok(query))
        return [self.chunks[i] for i in sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]]

    def search(self, query: str, k: int = 4) -> list[dict]:
        pos = {id(c): i for i, c in enumerate(self.chunks)}
        return sorted(self.ranked(query, k), key=lambda c: pos[id(c)])

