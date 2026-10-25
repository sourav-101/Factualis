from __future__ import annotations

import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
INCLUDE = ["factualis", "pipeline", "web", "configs", "data", "tests", "README.md", "requirements.txt"]
SKIP_SUFFIXES = {".bak", ".pyc"}
SKIP_NAMES = {"__pycache__"}


def main() -> None:
    out = ROOT / "dist" / "factualis_colab.zip"
    out.parent.mkdir(exist_ok=True)
    n = 0
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
        for item in INCLUDE:
            p = ROOT / item
            files = [p] if p.is_file() else sorted(f for f in p.rglob("*") if f.is_file())
            for f in files:
                if f.suffix in SKIP_SUFFIXES or SKIP_NAMES & set(f.parts) or f.name.endswith(".review.md"):
                    continue
                z.write(f, f.relative_to(ROOT).as_posix())
                n += 1
    print(f"wrote {out} ({n} files, {out.stat().st_size / 1e6:.1f} MB) - upload it to MyDrive/Factualis/")


if __name__ == "__main__":
    main()
