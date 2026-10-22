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
        text = data if isinstance(data, str) else json.dumps(data, indent=2, ensure_ascii=False)
        blocks.append(
            f'<section id="part{index}"><h2>{html.escape(heading)}</h2>'
            f'<pre>{html.escape(text)}</pre></section>'
        )
    page = (
        '<!doctype html><html lang="en"><meta charset="utf-8">'
        f"<title>{html.escape(title)}</title>"
        "<style>body{font:16px/1.5 system-ui;background:#edf1f5;color:#182336;margin:0}"
        "main{max-width:1080px;margin:24px auto}section,header{background:white;padding:24px;"
        "margin:18px 0;border-radius:10px}pre{white-space:pre-wrap;overflow-wrap:anywhere;"
        "font:14px/1.5 monospace;background:#142338;color:#e8eef5;padding:20px}"
        "h1,h2{margin-top:0}</style><main><header>"
        f"<h1>{html.escape(title)}</h1>"
        "<p>Actual saved experimental records. Post-run viewer, not a live terminal.</p>"
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
