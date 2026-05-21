from __future__ import annotations

import argparse
import importlib.util
import importlib.metadata
import platform
import re
import sys


BLACKWELL_MIN_CUDA_CAPABILITY = (10, 0)
BLACKWELL_MIN_CUDA_RUNTIME = (12, 8)


def has_module(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


def dist_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def base_version(version: str | None) -> str | None:
    if version is None:
        return None
    return version.split("+", 1)[0]


def version_tuple(version: str | None) -> tuple[int, ...]:
    if version is None:
        return ()
    return tuple(int(part) for part in re.findall(r"\d+", base_version(version) or ""))


def require_base_version(
    package: str,
    installed: str | None,
    expected: str,
    errors: list[str],
) -> None:
    if base_version(installed) != expected:
        errors.append(f"{package} must be {expected} for this compatibility profile; got {installed or 'missing'}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("env", choices=["preprocess", "model"])
    parser.add_argument("--sequence", action="store_true", help="Also require OpenNMT sequence-generation dependencies.")
    args = parser.parse_args()

    print(f"python: {sys.version.split()[0]}")
    print(f"platform: {platform.platform()}")

    required = ["rxngraphormer"]
    if args.env == "preprocess":
        required += [
            "rxngraphormer.preprocess",
            "torch",
            "torch_geometric",
            "pandas",
            "dgl",
            "dgllife",
            "rdkit",
            "rxnmapper",
            "localmapper",
        ]
    else:
        required += ["rxngraphormer", "torch", "torch_geometric", "pandas", "rdkit", "sklearn", "safetensors"]
        if args.sequence:
            required += ["onmt"]

    missing = []
    for module in required:
        present = has_module(module)
        print(f"{module}: {'ok' if present else 'missing'}")
        if not present:
            missing.append(module)

    if has_module("torch"):
        import torch

        print(f"torch: {torch.__version__}")
        print(f"cuda: {torch.version.cuda}")
        print(f"cuda_available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"gpu: {torch.cuda.get_device_name(0)}")
            print(f"capability: {torch.cuda.get_device_capability(0)}")

    if has_module("torch_geometric"):
        import torch_geometric

        print(f"torch_geometric: {torch_geometric.__version__}")

    dgl_version = dist_version("dgl")
    if dgl_version is not None:
        print(f"dgl: {dgl_version}")

    policy_errors: list[str] = []
    if has_module("torch"):
        import torch

        if args.env == "preprocess":
            print(
                "profile: preprocessing plus atom mapping; Torch, DGL, and CUDA "
                "wheel selection are resolved outside package metadata"
            )
            if torch.cuda.is_available():
                print(
                    "note: preprocessing and atom mapping do not require CUDA; "
                    "if a GPU wheel causes trouble on newer hardware, hide CUDA devices "
                    "or run the preprocessing stage in a separate older environment."
                )
        elif torch.cuda.is_available():
            capability = torch.cuda.get_device_capability(0)
            cuda_runtime = version_tuple(torch.version.cuda)
            if (
                capability >= BLACKWELL_MIN_CUDA_CAPABILITY
                and cuda_runtime < BLACKWELL_MIN_CUDA_RUNTIME
            ):
                policy_errors.append(
                    "model environment detected a Blackwell-class GPU but the torch CUDA "
                    f"runtime is {torch.version.cuda}. Use a CUDA 12.8+ PyTorch wheel for "
                    "the model stage and provide preprocessed dataset artifacts."
                )

    if missing:
        raise SystemExit(f"missing modules for {args.env}: {', '.join(missing)}")
    if policy_errors:
        raise SystemExit("\n".join(policy_errors))


if __name__ == "__main__":
    main()
