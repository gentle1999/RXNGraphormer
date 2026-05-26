from __future__ import annotations

import argparse
from pathlib import Path

from rxngraphormer.serialization import convert_serialized_file, prefer_safetensors_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert RXNGraphormer .pt/.pth/.ckpt artifacts to .safetensors."
    )
    parser.add_argument("input", type=Path, help="Input .pt/.pth/.ckpt file.")
    parser.add_argument(
        "output",
        nargs="?",
        type=Path,
        help="Output .safetensors file. Defaults to the input path with a .safetensors suffix.",
    )
    parser.add_argument(
        "--kind",
        choices=["checkpoint", "processed-data"],
        default="checkpoint",
        help="Artifact type to convert.",
    )
    parser.add_argument("--map-location", default="cpu")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    output = args.output or prefer_safetensors_path(args.input)
    if output.exists() and not args.overwrite:
        raise SystemExit(f"Output already exists: {output}. Pass --overwrite to replace it.")

    converted = convert_serialized_file(
        args.input,
        output,
        kind=args.kind,
        map_location=args.map_location,
    )
    print(converted)


if __name__ == "__main__":
    main()
