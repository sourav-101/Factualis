from __future__ import annotations

import argparse
import base64
import shutil
import subprocess
import tempfile
from pathlib import Path

DOCS = Path("docs")
BROWSERS = [r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
            r"C:\Program Files\Microsoft\Edge\Application\msedge.exe",
            r"C:\Program Files\Google\Chrome\Application\chrome.exe", "msedge", "google-chrome", "chromium"]


CAPTIONS = {
    "01_home": "The Factualis home page with a question entered.",
    "02_answered": "An answer grounded in the Wire Transfer SOP; both claims are found in the policy text.",
    "03_view_in_document": "View in document: the cited clause with the quotation highlighted.",
    "04_not_in_policy": "A trap question about crypto transfers, answered as not in policy.",
    "05_verified_mode": "Verified mode switched on, with the six documents of the policy library listed.",
    "06_structured_output_off": "Structured output switched off: the raw model output cannot be checked.",
}
FIRST_SCREENSHOT_FIGURE = 9


def _screenshots() -> str:
    shots = sorted(p for p in (DOCS / "screenshots").glob("*") if p.suffix.lower() in {".png", ".jpg", ".jpeg"})
    if not shots:
        return ""
    figs = []
    for n, p in enumerate(shots, FIRST_SCREENSHOT_FIGURE):
        mime = "image/png" if p.suffix.lower() == ".png" else "image/jpeg"
        data = base64.b64encode(p.read_bytes()).decode()
        caption = CAPTIONS.get(p.stem, p.stem.split("_", 1)[-1].replace("_", " "))
        figs.append(f'<figure><img src="data:{mime};base64,{data}" style="width:100%;border:.6pt solid #000;display:block">'
                    f'<figcaption><b>Figure {n}:</b> {caption}</figcaption></figure>')
    rows = ["".join(figs[i:i + 2]) for i in range(0, len(figs), 2)]
    return "".join(f'<div class="pair">{r}</div>' for r in rows)


def _browser(explicit: str | None) -> str:
    for b in ([explicit] if explicit else BROWSERS):
        if b and (Path(b).exists() or shutil.which(b)):
            return b
    raise SystemExit("no Edge/Chrome found; pass --edge PATH")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default="https://github.com/sourav-101/Factualis")
    ap.add_argument("--edge", default=None)
    ap.add_argument("--out", type=Path, default=DOCS / "Factualis_Technical_Report.pdf")
    a = ap.parse_args()

    html = (DOCS / "report.html").read_text(encoding="utf-8")
    for key, val in {"__REPO_LINK__": f'<a href="{a.repo}" style="color:inherit">{a.repo.removeprefix("https://")}</a>',
                     "__SCREENSHOTS__": _screenshots()}.items():
        html = html.replace(key, val)

    with tempfile.TemporaryDirectory() as tmp:
        page = Path(tmp) / "report.html"
        page.write_text(html, encoding="utf-8")
        out = a.out.resolve()
        out.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run([_browser(a.edge), "--headless=new", "--disable-gpu", "--no-pdf-header-footer",
                        f"--user-data-dir={Path(tmp) / 'profile'}", "--virtual-time-budget=8000",
                        f"--print-to-pdf={out}", page.as_uri()], check=True, timeout=180)
    print(f"wrote {out} ({out.stat().st_size // 1024} KB)")


if __name__ == "__main__":
    main()
