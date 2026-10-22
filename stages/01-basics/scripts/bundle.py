"""Package explicitly named experiment artifacts with hashes and a required notice."""

import argparse
import json
import zipfile
from pathlib import Path

from aml.data import sha256


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, nargs="+", required=True)
    parser.add_argument("--notice", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = Path.cwd().resolve()
    files = []
    for source in [args.notice, *args.input]:
        if not source.exists():
            raise FileNotFoundError(source)
        for path in sorted(source.rglob("*")) if source.is_dir() else [source]:
            if path.is_file():
                relative = path.resolve().relative_to(root)
                if any(part in (".venv", ".git", "__pycache__") for part in relative.parts):
                    raise ValueError(f"Not an experiment artifact: {relative}")
                if path.stat().st_size > 20_000_000:
                    raise ValueError(f"Use a GitHub release for large checkpoint: {relative}")
                files.append(relative)
    files = sorted(set(files))
    if args.output.resolve() in [root / path for path in files]:
        raise ValueError("Bundle cannot include itself")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    checksums = {str(path): sha256(path) for path in files}
    with zipfile.ZipFile(args.output, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in files:
            archive.write(path)
        archive.writestr("SHA256.json", json.dumps(checksums, indent=2) + "\n")
    with zipfile.ZipFile(args.output) as archive:
        if archive.testzip() is not None:
            raise ValueError("Generated bundle failed its CRC check")
    print(json.dumps({"bundle": str(args.output), "bytes": args.output.stat().st_size,
                      "sha256": sha256(args.output), "files": len(files)}))


if __name__ == "__main__":
    main()
