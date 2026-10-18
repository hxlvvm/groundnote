"""The checked-note page: note clauses underlined by verdict, transcript beside it.

Standard library only, so the browser demo can run it in Pyodide next to check.py.
"""
import html

from groundnote.check import check_note

COLOR = {"supported": "#1a7f37", "contradicted": "#cf222e", "unsupported": "#9a6700", "unclear": "#6e7781"}
LABEL = {"supported": "words found in transcript", "contradicted": "contradicts transcript",
         "unsupported": "not found in transcript", "unclear": "unclear"}
BANNER = ('<p style="border-left:3px solid #9a6700;padding:4px 12px"><b>Machine-generated draft from a research prototype. '
          "Not for clinical use.</b> Green means the clause's words were found in the transcript, not that the clause is "
          "correct.</p>")


def render_html(sections, turns):
    rows = []
    for h, text in sections:
        rows.append(f"<h3>{html.escape(h)}</h3><p>")
        for c in check_note(text, turns):
            ev = "; ".join(f"[{i + 1}] {turns[i][0]}: {turns[i][1]}" for i in c.evidence if i is not None)
            rows.append(f'<span style="border-bottom:2px solid {COLOR[c.verdict]}" title="{LABEL[c.verdict]}: '
                        f'{html.escape(ev, quote=True)}">{html.escape(c.text)}</span> ')
        rows.append("</p>")
    dia = "".join(f"<li><b>{html.escape(s)}</b>: {html.escape(u)}</li>" for s, u in turns)
    legend = " ".join(f'<span style="color:{v}">&#9632; {LABEL[k]}</span>' for k, v in COLOR.items())
    return (f"<!doctype html><meta charset=utf-8><title>groundnote</title><body style='font:15px system-ui;"
            f"max-width:1100px;margin:auto;padding:16px;display:grid;grid-template-columns:1fr 1fr;gap:24px'>"
            f"<div><h2>Note</h2>{BANNER}<p>{legend}<br><small>Hover a clause for its evidence turn.</small></p>{''.join(rows)}</div>"
            f"<div><h2>Transcript</h2><ol>{dia}</ol></div></body>")
