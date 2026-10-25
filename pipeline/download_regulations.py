from __future__ import annotations

import hashlib
import http.client
import json
import urllib.request
from pathlib import Path

OUT = Path("data/regulations")

SOURCES = {
    "rbi_kyc_master_direction": [
        "https://rbidocs.rbi.org.in/rdocs/notification/PDFs/MD18KYCF6E92C82E1E1419D87323E3869BC9F13.PDF",
        "https://website.rbi.org.in/documents/87730/39872377/MD18KYCF6E92C82E1E1419D87323E3869BC9F13.pdf",
    ],
    "pml_maintenance_of_records_rules": [
        "https://dor.gov.in/files/rules_files/2005.pdf",
    ],
    "sebi_pit_regulations": [
        "https://www.sebi.gov.in/sebi_data/attachdocs/aug-2026/1786963663321.pdf",
    ],
}


def fetch(url: str, attempts: int = 4) -> bytes:
    data = b""
    for _ in range(attempts):
        headers = {"User-Agent": "Mozilla/5.0 (Factualis research)"}
        if data:
            headers["Range"] = f"bytes={len(data)}-"
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=headers), timeout=120) as r:
                total = int(r.headers.get("Content-Length") or 0) + len(data)
                data += r.read()
                if not total or len(data) >= total:
                    return data
        except http.client.IncompleteRead as e:
            data += e.partial
    raise IOError(f"incomplete download ({len(data)} bytes)")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    manifest = {}
    for doc_id, urls in SOURCES.items():
        dest = OUT / f"{doc_id}.pdf"
        if dest.exists() and dest.read_bytes().startswith(b"%PDF"):
            data = dest.read_bytes()
            manifest[doc_id] = {"url": f"existing file (expected source: {urls[0]})", "bytes": len(data),
                                "sha256": hashlib.sha256(data).hexdigest()}
            print(f"ok {doc_id}: already present")
            continue
        for url in urls:
            try:
                data = fetch(url)
            except Exception as e:
                print(f"  {doc_id}: {url} failed ({e})")
                continue
            if not data.startswith(b"%PDF"):
                print(f"  {doc_id}: {url} did not return a PDF (bot check / HTML page)")
                continue
            dest.write_bytes(data)
            manifest[doc_id] = {"url": url, "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}
            print(f"ok {doc_id}: {len(data) / 1e6:.2f} MB <- {url}")
            break
        else:
            print(f"!! {doc_id}: all sources failed - download it manually into {dest}")
    (OUT / "SOURCES.json").write_text(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
