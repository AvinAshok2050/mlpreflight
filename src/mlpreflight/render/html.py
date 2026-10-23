"""Standalone HTML report - no external assets, safe to email or attach to CI."""
from __future__ import annotations

import html as _html

from ..core.finding import Severity

_COLOR = {"critical": "#c0392b", "warning": "#d68910",
          "passed": "#1e8449", "skipped": "#7f8c8d"}

_CSS = """
body{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,sans-serif;
margin:0;padding:32px;background:#f5f6f7;color:#1c1e21;line-height:1.55}
.wrap{max-width:960px;margin:0 auto}
h1{font-size:22px;margin:0 0 4px}
.sub{color:#65676b;font-size:13px;margin-bottom:24px}
.score{display:flex;align-items:center;gap:24px;background:#fff;border-radius:10px;
padding:24px;margin-bottom:20px;box-shadow:0 1px 3px rgba(0,0,0,.08)}
.num{font-size:52px;font-weight:700;line-height:1}
.verdict{font-weight:600;font-size:15px}
.pills{display:flex;gap:8px;margin-top:8px;flex-wrap:wrap}
.pill{padding:3px 10px;border-radius:12px;font-size:12px;color:#fff;font-weight:600}
.card{background:#fff;border-radius:10px;padding:18px 20px;margin-bottom:12px;
box-shadow:0 1px 3px rgba(0,0,0,.08);border-left:4px solid #ddd}
.card h3{margin:0 0 6px;font-size:15px}
.meta{font-size:11px;color:#8a8d91;text-transform:uppercase;letter-spacing:.4px}
.lbl{font-weight:600;font-size:12px;color:#65676b;margin-top:10px}
table{width:100%;border-collapse:collapse;margin-top:10px;font-size:13px}
th,td{text-align:left;padding:6px 8px;border-bottom:1px solid #eee}
th{background:#fafafa;font-size:11px;text-transform:uppercase;color:#65676b}
code{background:#f0f2f5;padding:1px 5px;border-radius:3px;font-size:12px}
.note{background:#fff8e1;border-left:4px solid #f0b429;padding:12px 16px;
border-radius:6px;font-size:13px;margin-bottom:20px}
"""


def render_html(report) -> str:
    m = report.meta
    score_color = ("#c0392b" if report.has_blocking_issues
                   else "#1e8449" if report.score >= 90 else "#d68910")

    pills = "".join(
        f'<span class="pill" style="background:{_COLOR[k]}">{v} {k}</span>'
        for k, v in [("critical", len(report.critical)), ("warning", len(report.warnings)),
                     ("passed", len(report.passed)), ("skipped", len(report.skipped))] if v
    )

    assumptions = ""
    if report.assumptions:
        items = "".join(f"<li>{_html.escape(a)}</li>" for a in report.assumptions)
        assumptions = f'<div class="note"><b>Assumptions made:</b><ul>{items}</ul></div>'

    cards = []
    order = [Severity.CRITICAL, Severity.WARNING, Severity.SKIPPED, Severity.PASSED]
    for sev in order:
        for f in [x for x in report.findings if x.severity is sev]:
            rows = "".join(
                f"<tr><td><code>{_html.escape(ev.column)}</code></td>"
                f"<td>{_html.escape(ev.statistic)}</td>"
                f"<td>{ev.value:.4f}</td>"
                f"<td>{'' if ev.threshold is None else ev.threshold}</td></tr>"
                for ev in f.columns
            )
            table = (f"<table><tr><th>Column</th><th>Statistic</th><th>Value</th>"
                     f"<th>Threshold</th></tr>{rows}</table>" if rows else "")
            body = ""
            if sev in (Severity.CRITICAL, Severity.WARNING):
                body = (f'<div class="lbl">Why it matters</div>{_html.escape(f.why)}'
                        f'<div class="lbl">Recommended action</div>'
                        f'{_html.escape(f.recommendation)}')
            conf = (f'<div class="lbl">Confidence: {f.confidence:.0%}</div>'
                    if f.confidence < 1 else "")
            cards.append(
                f'<div class="card" style="border-left-color:{_COLOR[sev.value]}">'
                f'<div class="meta">{_html.escape(f.category)} &middot; {f.check_id}</div>'
                f'<h3>{_html.escape(f.title)}</h3>'
                f'{_html.escape(f.what)}{body}{conf}{table}</div>'
            )

    return f"""<!DOCTYPE html><html><head><meta charset="utf-8">
<title>mlpreflight Report</title><style>{_CSS}</style></head><body><div class="wrap">
<h1>mlpreflight Report</h1>
<div class="sub">{m.get('n_rows', 0):,} rows &times; {m.get('n_cols', 0)} columns
&middot; target <code>{_html.escape(str(m.get('target', '')))}</code>
&middot; {m.get('n_classes', '?')} classes</div>
<div class="score"><div class="num" style="color:{score_color}">{report.score}</div>
<div><div class="verdict">{_html.escape(report.verdict)}</div>
<div class="pills">{pills}</div></div></div>
{assumptions}{''.join(cards)}
</div></body></html>"""
