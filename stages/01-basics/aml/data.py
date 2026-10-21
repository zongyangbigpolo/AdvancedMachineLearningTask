"""Stream line-delimited UTF-8 documents to portable little-endian uint32 tokens."""

import argparse
import hashlib
import json
import mmap
import struct
import sys
from array import array
from pathlib import Path

import torch

from aml.tokenizer import ByteBPE


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def encode_file(source: Path, destination: Path, tokenizer: ByteBPE) -> dict:
    documents = tokens = 0
    with source.open(encoding="utf-8") as reader, destination.open("wb") as writer:
        for line in reader:
            text = line.rstrip("\r\n")
            if not text.strip():
                continue
            ids = tokenizer.encode(text, add_eos=True)
            writer.write(struct.pack(f"<{len(ids)}I", *ids))
            documents += 1
            tokens += len(ids)
    if not tokens:
        raise ValueError(f"No nonempty documents in {source}")
    return {
        "source": str(source.resolve()),
        "source_sha256": sha256(source),
        "token_sha256": sha256(destination),
        "documents": documents,
        "tokens": tokens,
    }


class TokenStream:
    def __init__(self, path: Path):
        if path.stat().st_size % 4 or path.stat().st_size < 8:
            raise ValueError(f"Invalid uint32 token stream: {path}")
        self.file = path.open("rb")
        self.mapping = mmap.mmap(self.file.fileno(), 0, access=mmap.ACCESS_READ)
        self.length = len(self.mapping) // 4

    def close(self) -> None:
        self.mapping.close()
        self.file.close()

    def batch(self, batch_size: int, context: int, generator: torch.Generator, device: str):
        if self.length <= context:
            raise ValueError(f"Need at least {context + 1} tokens; have {self.length}")
        starts = torch.randint(self.length - context, (batch_size,), generator=generator)
        rows = []
        for start in starts.tolist():
            values = array("I")
            values.frombytes(self.mapping[start * 4 : (start + context + 1) * 4])
            if sys.byteorder != "little":
                values.byteswap()
            rows.append(list(values))
        batch = torch.tensor(rows, dtype=torch.long, device=device)
        return batch[:, :-1], batch[:, 1:]


def main() -> None:
    parser = argparse.ArgumentParser(description="Encode distinct train and validation documents")
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--val", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.train.resolve() == args.val.resolve() or sha256(args.train) == sha256(args.val):
        parser.error("Train and validation inputs must be distinct")
    if args.output.exists() and any(args.output.iterdir()):
        parser.error("Output must be new or empty; do not overwrite an existing dataset")
    args.output.mkdir(parents=True, exist_ok=True)
    tokenizer = ByteBPE.load(args.tokenizer)
    tokenizer.save(args.output / "tokenizer.json")
    metadata = {
        "format": "uint32-le-v1",
        "tokenizer_sha256": sha256(args.output / "tokenizer.json"),
        "vocab_size": tokenizer.vocab_size,
        "eos_id": tokenizer.eos_id,
        "train": encode_file(args.train, args.output / "train.bin", tokenizer),
        "val": encode_file(args.val, args.output / "val.bin", tokenizer),
    }
    (args.output / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
