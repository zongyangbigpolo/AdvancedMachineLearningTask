"""Save machine-readable results and an HTML view of the same measurements."""

import html
import json
import platform
import sys
from pathlib import Path

import torch


def save_record(directory: Path, title: str, result: dict, sections: dict[str, object]) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    record = {
        "command": " ".join(sys.argv),
        "python": platform.python_version(),
        "torch": str(torch.__version__),
        "platform": platform.platform(),
        **result,
    }
    (directory / "results.json").write_text(json.dumps(record, indent=2, ensure_ascii=False) + "\n")
    blocks = []
    for index, (heading, data) in enumerate(sections.items()):
        if isinstance(data, list) and data and all(isinstance(row, dict) for row in data):
            columns = list(data[0])
            cells = []
            for row in data:
                cells.append("<tr>" + "".join(
                    f"<td>{html.escape(str(row.get(column, '')))}</td>" for column in columns
                ) + "</tr>")
            content = ("<table><thead><tr>" +
                       "".join(f"<th>{html.escape(column)}</th>" for column in columns) +
                       "</tr></thead><tbody>" + "".join(cells) + "</tbody></table>")
        else:
            text = data if isinstance(data, str) else json.dumps(data, indent=2, ensure_ascii=False)
            content = f"<pre>{html.escape(text)}</pre>"
        blocks.append(f'<section id="part{index}"><h2>{html.escape(heading)}</h2>{content}</section>')
    page = (
        '<!doctype html><html lang="en"><meta charset="utf-8">'
        f"<title>{html.escape(title)}</title>"
        "<style>body{font:16px/1.5 system-ui;background:#edf1f5;color:#182336;margin:0}"
        "main{max-width:1080px;margin:24px auto}section,header{background:white;padding:24px;"
        "margin:18px 0;border-radius:10px}pre{white-space:pre-wrap;overflow-wrap:anywhere;"
        "font:14px/1.5 monospace;background:#142338;color:#e8eef5;padding:20px}"
        "h1,h2{margin-top:0}table{border-collapse:collapse;width:100%;font-size:13px}"
        "th,td{border:1px solid #cbd5e1;padding:8px;text-align:left;overflow-wrap:anywhere}"
        "th{background:#e8eef5}</style><main><header>"
        f"<h1>{html.escape(title)}</h1>"
        "<p>Experiment configuration, measurements and results.</p>"
        f"<pre>{html.escape(record['command'])}</pre>"
        '<a href="results.json">Machine-readable results</a></header>'
        + ('<section id="curve"><h2>Measured validation loss</h2><img src="loss.svg" '
           'alt="Validation loss from saved metrics" style="width:100%"></section>'
           if (directory / "loss.svg").exists() else "")
        + "".join(blocks)
        + "</main></html>"
    )
    (directory / "index.html").write_text(page, encoding="utf-8")


def loss_svg(points: list[tuple[int, float]], path: Path) -> None:
    if not points:
        raise ValueError("A loss chart needs measurements")
    low = min(value for _, value in points)
    high = max(value for _, value in points)
    maximum = max(step for step, _ in points) or 1
    coordinates = " ".join(
        f"{50 + 850 * step / maximum:.2f},{300 - 250 * (value - low) / max(high - low, 1e-9):.2f}"
        for step, value in points
    )
    path.write_text(
        '<svg xmlns="http://www.w3.org/2000/svg" width="960" height="360">'
        '<rect width="960" height="360" fill="white"/>'
        '<path d="M50 30 V300 H920" fill="none" stroke="#333"/>'
        f'<polyline points="{coordinates}" fill="none" stroke="#2469a8" stroke-width="3"/>'
        f'<text x="50" y="22">Validation CE: {high:.4f} to {low:.4f}</text>'
        f'<text x="760" y="335">Updates: {maximum}</text></svg>'
    )
