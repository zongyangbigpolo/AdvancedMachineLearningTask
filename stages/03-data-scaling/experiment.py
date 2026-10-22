"""Clean a MediaWiki XML export, train size pilots, and test an extrapolation."""

import argparse
import hashlib
import html
import json
import math
import re
import time
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from collections import Counter
from dataclasses import asdict
from html.parser import HTMLParser
from pathlib import Path

import torch

from aml.data import TokenStream, encode_file, sha256
from aml.model import ModelConfig, TransformerLM, cross_entropy
from aml.recording import save_record
from aml.tokenizer import ByteBPE


TITLES = ("Cat", "Dog", "Tree", "Water", "Sun", "Moon", "Earth", "River",
          "Mountain", "Bird", "Fish", "Book", "School", "Music", "Computer", "Electricity")


class PlainHTML(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []

    def handle_data(self, data: str) -> None:
        self.parts.append(data)


def without_templates(text: str) -> str:
    depth = 0
    result = []
    index = 0
    while index < len(text):
        pair = text[index:index + 2]
        if pair == "{{":
            depth += 1
            index += 2
        elif pair == "}}" and depth:
            depth -= 1
            index += 2
        else:
            if not depth:
                result.append(text[index])
            index += 1
    if depth:
        raise ValueError("Unbalanced template braces")
    return "".join(result)


def clean(text: str) -> str:
    text = re.split(r"(?im)^==+\s*(references|other websites|sources|notes)\s*==+", text)[0]
    text = re.sub(r"<!--.*?-->", " ", text, flags=re.S)
    text = re.sub(r"<ref\b[^>]*>.*?</ref\s*>|<ref\b[^>]*/>", " ", text, flags=re.S | re.I)
    text = without_templates(text)
    text = re.sub(r"\{\|.*?\|\}", " ", text, flags=re.S)
    text = re.sub(r"\[\[(?:File|Image|Category):[^\]]*\]\]", " ", text, flags=re.I)
    text = re.sub(r"\[\[([^\[\]]+)\]\]", lambda match: match[1].split("|")[-1], text)
    text = re.sub(r"\[https?://\S+(?:\s+([^\]]*))?\]", lambda match: match[1] or " ", text)
    text = re.sub(r"'{2,}|={2,}|^\s*[*#:;]+", " ", text, flags=re.M)
    parser = PlainHTML()
    parser.feed(text)
    return " ".join(html.unescape(" ".join(parser.parts)).split())


def shingles(text: str) -> set[tuple[str, ...]]:
    words = re.findall(r"\w+", text.casefold())
    return {tuple(words[index:index + 5]) for index in range(max(1, len(words) - 4))}


def prepare(directory: Path, source_xml: Path | None = None) -> tuple[dict, Path]:
    directory.mkdir(parents=True, exist_ok=True)
    url = "https://simple.wikipedia.org/wiki/Special:Export?" + urllib.parse.urlencode({
        "pages": "\n".join(TITLES), "curonly": "1",
    })
    if source_xml is None:
        request = urllib.request.Request(url, headers={"User-Agent": "AML-learning-lab/0.1 (educational XML export)"})
        with urllib.request.urlopen(request, timeout=90) as response:
            raw = response.read(10_000_001)
    else:
        with source_xml.open("rb") as source:
            raw = source.read(10_000_001)
    if len(raw) > 10_000_000:
        raise ValueError("XML export exceeds the declared 10 MB download budget")
    (directory / "export.xml").write_bytes(raw)
    root = ET.fromstring(raw)
    namespace = {"m": root.tag.partition("}")[0].lstrip("{")}
    reasons: Counter[str] = Counter()
    kept = []
    seen: set[str] = set()
    near: list[set[tuple[str, ...]]] = []
    source = []
    pages = root.findall("m:page", namespace)
    if len(pages) != len(TITLES):
        raise ValueError(f"Expected {len(TITLES)} exported articles, received {len(pages)}")
    for page in pages:
        title = page.findtext("m:title", namespaces=namespace)
        revision = page.findtext("m:revision/m:id", namespaces=namespace)
        text = page.findtext("m:revision/m:text", namespaces=namespace)
        if title is None or revision is None or text is None:
            raise ValueError("Missing title, revision, or text in XML export")
        source.append({"title": title, "revision": revision,
                       "url": "https://simple.wikipedia.org/w/index.php?oldid=" + revision})
        if page.find("m:redirect", namespace) is not None:
            reasons["redirect"] += 1
            continue
        try:
            text = clean(text)
        except ValueError as error:
            if str(error) != "Unbalanced template braces":
                raise
            reasons["unbalanced_template"] += 1
            continue
        if len(text) < 120:
            reasons["too_short"] += 1
            continue
        if sum(character.isalpha() for character in text) / len(text) < 0.5:
            reasons["low_alphabetic_ratio"] += 1
            continue
        digest = hashlib.sha256(text.casefold().encode()).hexdigest()
        if digest in seen:
            reasons["exact_duplicate"] += 1
            continue
        signature = shingles(text)
        if any(len(signature & other) / len(signature | other) >= 0.8 for other in near):
            reasons["near_duplicate"] += 1
            continue
        seen.add(digest)
        near.append(signature)
        kept.append((title, text, digest))
    kept.sort()
    train = [item for index, item in enumerate(kept) if index % 4 != 0]
    val = [item for index, item in enumerate(kept) if index % 4 == 0]
    if len(train) < 4 or len(val) < 2:
        raise ValueError("Too few cleaned articles for a held-out evaluation")
    for name, rows in (("train", train), ("val", val)):
        (directory / f"{name}.txt").write_text("\n".join(row[1] for row in rows) + "\n")
    overlap = {row[2] for row in train} & {row[2] for row in val}
    if overlap:
        raise AssertionError("Cross-split exact duplicate")
    card = {
        "source": url, "raw_sha256": hashlib.sha256(raw).hexdigest(), "raw_bytes": len(raw),
        "format": "MediaWiki XML export, latest revision of 16 named articles; not a full Wikipedia dump",
        "license": "CC BY-SA 4.0", "license_url": "https://creativecommons.org/licenses/by-sa/4.0/",
        "attribution": "Simple English Wikipedia contributors; per-article revision links below",
        "changes": "Removed templates/references/HTML/links, normalized whitespace, filtered and split by article",
        "articles": source, "input_documents": len(pages), "kept_documents": len(kept),
        "removed": dict(reasons), "train_documents": len(train), "validation_documents": len(val),
        "near_duplicate_threshold": 0.8, "shingle_words": 5, "cross_split_exact_duplicates": len(overlap),
        "train_sha256": sha256(directory / "train.txt"), "validation_sha256": sha256(directory / "val.txt"),
    }
    tokenizer = ByteBPE.train((row[1] for row in train), 512)
    tokenizer.save(directory / "tokenizer.json")
    for name in ("train", "val"):
        card[name + "_encoding"] = encode_file(directory / f"{name}.txt", directory / f"{name}.bin", tokenizer)
    (directory / "source.json").write_text(json.dumps(card, indent=2) + "\n")
    return card, directory


@torch.no_grad()
def evaluate(model: TransformerLM, stream: TokenStream) -> float:
    model.eval()
    generator = torch.Generator().manual_seed(900)
    losses = []
    for _ in range(8):
        x, y = stream.batch(4, model.config.context_length, generator, "cpu")
        losses.append(cross_entropy(model(x), y).item())
    model.train()
    return sum(losses) / len(losses)


def pilot(width: int, seed: int, steps: int, data: Path, output: Path) -> dict:
    torch.manual_seed(seed)
    tokenizer = ByteBPE.load(data / "tokenizer.json")
    config = ModelConfig(tokenizer.vocab_size, 64, width, 2, 4, width * 3)
    model = TransformerLM(config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.1)
    generator = torch.Generator().manual_seed(seed)
    train, val = TokenStream(data / "train.bin"), TokenStream(data / "val.bin")
    began = time.perf_counter()
    losses = []
    try:
        initial = evaluate(model, val)
        for step in range(steps):
            x, y = train.batch(8, config.context_length, generator, "cpu")
            loss = cross_entropy(model(x), y)
            if not torch.isfinite(loss):
                raise FloatingPointError("Non-finite pilot loss")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1, error_if_nonfinite=True)
            optimizer.step()
            if (step + 1) % 25 == 0 or step + 1 == steps:
                losses.append({"step": step + 1, "validation_ce": evaluate(model, val)})
        final = evaluate(model, val)
        checkpoint = output / f"width-{width}-seed-{seed}.pt"
        torch.save({"config": {"model": asdict(config)}, "model": model.state_dict(), "step": steps,
                    "device": "cpu", "tokenizer_sha256": sha256(data / "tokenizer.json")}, checkpoint)
        if width == 96 and seed == 42:
            (output / "holdout.pt").write_bytes(checkpoint.read_bytes())
            (output / "tokenizer.json").write_bytes((data / "tokenizer.json").read_bytes())
        return {"width": width, "seed": seed, "parameters": model.parameter_count(),
                "tokens_seen": steps * 8 * 64, "initial_ce": initial, "validation_ce": final,
                "seconds": time.perf_counter() - began, "checkpoint_sha256": sha256(checkpoint),
                "curve": losses}
    finally:
        train.close()
        val.close()


def fit_power(points: list[tuple[int, float]]) -> dict:
    if len(points) < 4:
        raise ValueError("Need at least four fitting points")
    n0 = min(size for size, _ in points)
    best = None
    for alpha_index in range(1, 201):
        alpha = alpha_index / 100
        xs = [(size / n0) ** -alpha for size, _ in points]
        for offset_index in range(100):
            offset = min(loss for _, loss in points) * offset_index / 100
            amplitude = sum(x * (point[1] - offset) for x, point in zip(xs, points)) / sum(x*x for x in xs)
            error = sum((offset + amplitude*x - point[1]) ** 2 for x, point in zip(xs, points))
            if best is None or error < best["squared_error"]:
                best = {"offset": offset, "amplitude": amplitude, "alpha": alpha,
                        "n0": n0, "squared_error": error}
    if best is None:
        raise AssertionError("Fit produced no candidate")
    return best


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--xml", type=Path, help="Replay a saved raw XML export instead of downloading")
    parser.add_argument("--output", type=Path, default=Path("runs/scaling"))
    args = parser.parse_args()
    if args.steps < 1:
        parser.error("--steps must be positive")
    torch.set_num_threads(1)
    output = args.output
    if output.exists() and any(output.iterdir()):
        raise ValueError("Choose a new --output directory; do not overwrite experiments")
    output.mkdir(parents=True)
    card, data = prepare(output / "data", args.xml)
    runs = []
    for width in (32, 48, 64, 80, 96):
        for seed in (42, 43, 44):
            measured = pilot(width, seed, args.steps, data, output)
            runs.append(measured)
            print(json.dumps(measured), flush=True)
    means = [
        (next(run["parameters"] for run in runs if run["width"] == width),
         sum(run["validation_ce"] for run in runs if run["width"] == width) / 3)
        for width in (32, 48, 64, 80, 96)
    ]
    fitted = fit_power(means[:-1])
    predicted = fitted["offset"] + fitted["amplitude"] * (means[-1][0] / fitted["n0"]) ** -fitted["alpha"]
    extrapolation = {
        "formula": "L(N) = offset + amplitude * (N/n0)^(-alpha)", "fit": fitted,
        "fit_points": means[:-1], "holdout_parameters": means[-1][0],
        "predicted_holdout_ce": predicted, "actual_holdout_ce": means[-1][1],
        "relative_error": abs(predicted - means[-1][1]) / means[-1][1],
        "protocol": "Same tokenizer, token budget, validation windows; three seeds; largest size excluded from fit",
    }
    result = {"data": card, "runs": runs, "extrapolation": extrapolation}
    report = Path("reports")
    save_record(report, "Stage 3 — XML cleaning and measured scaling extrapolation", result, {
        "1. XML export and cleaning summary": card,
        "2. Fifteen model-size training runs": [
            {key: run[key] for key in ("width", "seed", "parameters", "tokens_seen", "validation_ce", "seconds")}
            for run in runs
        ],
        "3. Largest-model hold-out prediction": extrapolation,
    })
    (report / "data-card.json").write_text(json.dumps(card, indent=2) + "\n")


if __name__ == "__main__":
    main()
