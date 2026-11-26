from . import config

CSS = """
@page { size: A4; margin: 20mm; }
body { font-family: "DejaVu Sans", "Noto Sans", sans-serif; font-size: 11pt; line-height: 1.45; color: #111; }
h1 { font-size: 18pt; margin-bottom: 6pt; }
h2 { font-size: 14pt; margin-top: 14pt; margin-bottom: 6pt; border-bottom: 1px solid #ccc; padding-bottom: 2pt; }
h3 { font-size: 12pt; margin-top: 10pt; margin-bottom: 4pt; }
h4 { font-size: 11pt; margin-top: 8pt; margin-bottom: 3pt; }
p { margin: 6pt 0; text-align: justify; }
table { border-collapse: collapse; width: 100%; font-size: 9pt; margin: 8pt 0; }
th, td { border: 1px solid #888; padding: 3px 5px; vertical-align: top; }
th { background: #eee; }
img { max-width: 100%; }
pre { font-size: 8pt; background: #f5f5f5; padding: 6px; }
code { font-size: 9pt; }
hr { border: none; border-top: 1px solid #ccc; margin: 12pt 0; }
"""


def markdown_to_html(md_text: str) -> str:
    import markdown

    html_body = markdown.markdown(md_text, extensions=["tables", "fenced_code"])
    return "<html><head><meta charset='utf-8'><style>" + CSS + "</style></head><body>" + html_body + "</body></html>"


def write_pdf_from_firefox(html_text: str, out_pdf) -> bool:
    import os
    import subprocess
    from pathlib import Path

    out_pdf = Path(out_pdf)
    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    tmp_html = out_pdf.parent / "report_tmp.html"
    tmp_html.write_text(html_text, encoding="utf-8")
    os.environ["MOZ_HEADLESS"] = "1"
    try:
        proc = subprocess.run(
            ["firefox", "--headless", "--print-to-pdf", str(out_pdf.resolve()), tmp_html.resolve().as_uri()],
            check=False,
            capture_output=True,
            timeout=120,
        )
        ok = proc.returncode == 0 and out_pdf.exists()
        return ok
    except Exception:
        return False
    finally:
        tmp_html.unlink(missing_ok=True)


def render(md_text: str, out_pdf, base_url: str = None) -> bool:
    html_text = markdown_to_html(md_text)
    try:
        from weasyprint import HTML

        HTML(string=html_text, base_url=base_url if base_url is not None else str(config.REPORT_DIR)).write_pdf(str(out_pdf))
        return True
    except Exception:
        return write_pdf_from_firefox(html_text, out_pdf)
