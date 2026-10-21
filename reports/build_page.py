"""Build the self-contained portfolio page and its PDF report from reports/page_template.html.

Charts referenced as <img src="outputs/...png"> are embedded as base64, so the page
works anywhere (GitHub download, a portfolio site, an email attachment) without the
outputs/ folder next to it. The PDF in docs/ is the same page printed by Chromium.

    .venv/bin/python reports/make_outputs.py     # charts first
    .venv/bin/python reports/build_page.py       # then the page and PDF
"""

from __future__ import annotations

import argparse
import base64
import os
import re
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
TEMPLATE = PROJECT_ROOT / "reports" / "page_template.html"
IMG_RE = re.compile(r'src="(outputs/[^"]+\.png)"')


def inline_images(html: str, root: Path) -> str:
    def repl(match: re.Match) -> str:
        data = base64.b64encode((root / match.group(1)).read_bytes()).decode()
        return f'src="data:image/png;base64,{data}"'

    return IMG_RE.sub(repl, html)


def print_pdf(html_path: Path, pdf_path: Path) -> None:
    from playwright.sync_api import sync_playwright

    with sync_playwright() as p:
        # CHROMIUM_PATH lets you point at an existing Chrome/Chromium instead of `playwright install`.
        exe = os.environ.get("CHROMIUM_PATH")
        browser = p.chromium.launch(executable_path=exe or None)
        page = browser.new_page()
        page.goto(html_path.resolve().as_uri(), wait_until="load")
        # The sticky nav makes no sense on paper.
        page.add_style_tag(content=".sticky-nav{display:none}")
        page.pdf(path=str(pdf_path), format="Letter", print_background=True,
                 margin={"top": "0.4in", "bottom": "0.5in", "left": "0.3in", "right": "0.3in"})
        browser.close()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--name", required=True, help="Output file stem, e.g. B2B_Lead_Funnel_Analysis")
    parser.add_argument("--no-pdf", action="store_true")
    args = parser.parse_args(argv)

    html = inline_images(TEMPLATE.read_text(encoding="utf-8"), PROJECT_ROOT)
    missing = IMG_RE.findall(html)
    if missing:
        raise SystemExit(f"Images not embedded: {missing}")
    page = PROJECT_ROOT / f"{args.name}.html"
    page.write_text(html, encoding="utf-8")
    print(f"wrote {page.name} ({page.stat().st_size / 1e6:.1f} MB)")
    if not args.no_pdf:
        pdf = PROJECT_ROOT / "docs" / f"{args.name}_Report.pdf"
        pdf.parent.mkdir(exist_ok=True)
        print_pdf(page, pdf)
        print(f"wrote docs/{pdf.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
