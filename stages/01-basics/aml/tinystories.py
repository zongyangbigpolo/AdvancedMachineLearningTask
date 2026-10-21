"""Fetch a bounded TinyStories teaching subset; no datasets/tokenizers library."""

import argparse
import hashlib
import json
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import urlopen


DATASET = "roneneldan/TinyStories"
LICENSE = "https://cdla.dev/sharing-1-0/"


def fetch_rows(split: str, count: int) -> tuple[list[str], list[dict]]:
    documents, pages = [], []
    for offset in range(0, count, 100):
        length = min(100, count - offset)
        query = urlencode({
            "dataset": DATASET, "config": "default", "split": split,
            "offset": offset, "length": length,
        })
        url = f"https://datasets-server.huggingface.co/rows?{query}"
        with urlopen(url, timeout=60) as response:
            raw = response.read()
        payload = json.loads(raw)
        rows = payload["rows"]
        if len(rows) != length:
            raise ValueError(f"Incomplete {split} page at {offset}")
        for index, row in enumerate(rows, start=offset):
            if row["row_idx"] != index or row["truncated_cells"]:
                raise ValueError(f"Truncated or reordered {split} row {index}")
            text = row["row"]["text"]
            if not isinstance(text, str) or not text.strip():
                raise ValueError(f"Empty or invalid {split} row {index}")
            documents.append(" ".join(text.split()))
        pages.append({"url": url, "sha256": hashlib.sha256(raw).hexdigest()})
    return documents, pages


def prepare(output: Path, train_rows: int, val_rows: int) -> dict:
    if not 1 <= train_rows <= 2000 or not 1 <= val_rows <= 500:
        raise ValueError("Teaching budget: 1..2000 train rows and 1..500 validation rows")
    if output.exists() and any(output.iterdir()):
        raise ValueError("Output must be new or empty")
    splits, provenance = {}, {}
    for split, count in (("train", train_rows), ("validation", val_rows)):
        documents, pages = fetch_rows(split, count)
        if len(set(documents)) != len(documents):
            raise ValueError(f"Duplicate documents in {split}; choose another subset")
        splits[split] = documents
        provenance[split] = pages
    if set(splits["train"]) & set(splits["validation"]):
        raise ValueError("Train/validation exact document overlap")
    output.mkdir(parents=True, exist_ok=True)
    files = {}
    for split, name in (("train", "train.txt"), ("validation", "val.txt")):
        content = "\n".join(splits[split]) + "\n"
        (output / name).write_text(content, encoding="utf-8")
        files[name] = {
            "documents": len(splits[split]),
            "sha256": hashlib.sha256(content.encode("utf-8")).hexdigest(),
        }
    manifest = {
        "dataset": DATASET, "authors": "Ronen Eldan and Yuanzhi Li",
        "source": f"https://huggingface.co/datasets/{DATASET}",
        "license": "CDLA-Sharing-1.0", "license_url": LICENSE,
        "selection": "First N rows of upstream train and validation splits; no random sampling",
        "modification": "Whitespace normalized; one complete story per line",
        "revision": "Live dataset-server responses, not revision-pinned; verify page and file hashes",
        "files": files, "pages": provenance,
    }
    (output / "source.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--train-rows", type=int, default=512)
    parser.add_argument("--val-rows", type=int, default=128)
    args = parser.parse_args()
    print(json.dumps(prepare(args.output, args.train_rows, args.val_rows), indent=2))


if __name__ == "__main__":
    main()
