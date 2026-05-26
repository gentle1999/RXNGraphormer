from __future__ import annotations

import argparse
import http.client
import json
import shutil
import subprocess
import time
import urllib.request
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ARTICLE_ID = "28356077"
FIGSHARE_API = f"https://api.figshare.com/v2/articles/{ARTICLE_ID}"

DATASET_FILES = {
    "50k_with_rxn_type.zip",
    "benmark_dataset.zip",
    "external_validation_dataset.zip",
    "OOS.zip",
    "pretrain.zip",
    "USPTO_50k.zip",
    "USPTO_480k.zip",
    "USPTO_full.zip",
    "USPTO_STEREO.zip",
}

MODEL_FILES = {
    "pretrained_classification_model.zip",
    "USPTO_50k_model.zip",
    "USPTO_480k_model.zip",
    "USPTO_full_model.zip",
    "USPTO_STEREO_model.zip",
    "buchwald_hartwig.zip",
    "C_H_func.zip",
    "suzuki_miyaura.zip",
    "thiol_addition.zip",
    "external_validation.zip",
}

PROFILE_FILES = {
    "datasets": DATASET_FILES,
    "models": MODEL_FILES,
    "bh-smoke": {"buchwald_hartwig.zip"},
    "uspto-smoke": {"USPTO_50k.zip", "USPTO_50k_model.zip"},
    "readme": DATASET_FILES | MODEL_FILES,
}


def request_for(url: str, *, start: int = 0) -> urllib.request.Request:
    headers = {"User-Agent": "rxngraphormer-reproduce/1.0"}
    if start > 0:
        headers["Range"] = f"bytes={start}-"
    return urllib.request.Request(url, headers=headers)


def urlopen_with_retry(url: str, timeout: int = 300, retries: int = 5, *, start: int = 0):
    last_error = None
    for attempt in range(1, retries + 1):
        try:
            return urllib.request.urlopen(request_for(url, start=start), timeout=timeout)
        except (OSError, http.client.HTTPException) as exc:
            last_error = exc
            if attempt == retries:
                break
            sleep_s = min(2 ** attempt, 30)
            print(f"retry {attempt}/{retries}: {url} failed with {exc}; sleeping {sleep_s}s")
            time.sleep(sleep_s)
    raise last_error


def fetch_manifest() -> dict[str, dict]:
    with urlopen_with_retry(FIGSHARE_API, timeout=60) as response:
        article = json.load(response)
    return {item["name"]: item for item in article["files"]}


def destination_for(filename: str, root: Path) -> Path:
    if filename in DATASET_FILES:
        return root / "dataset"
    if filename in MODEL_FILES:
        return root / "model_path"
    raise ValueError(f"Unknown asset destination for {filename}")


def download_file(
    url: str,
    path: Path,
    *,
    expected_size: int | None = None,
    chunk_size: int = 64 * 1024 * 1024,
    jobs: int = 8,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".part")
    if path.exists() and expected_size is not None and path.stat().st_size == expected_size:
        return

    start = tmp_path.stat().st_size if tmp_path.exists() else 0
    if expected_size is None or jobs <= 1:
        _download_range(url, tmp_path, start, None)
    else:
        download_ranges_parallel(url, tmp_path, path.name, start, expected_size, chunk_size, jobs)

    if path.suffix == ".zip" and not zipfile.is_zipfile(tmp_path):
        tmp_path.unlink(missing_ok=True)
        raise RuntimeError(f"Downloaded file is not a zip archive: {url}")
    tmp_path.replace(path)


def download_ranges_parallel(
    url: str,
    tmp_path: Path,
    display_name: str,
    start: int,
    expected_size: int,
    chunk_size: int,
    jobs: int,
) -> None:
    if start >= expected_size:
        return

    chunk_dir = tmp_path.with_suffix(tmp_path.suffix + ".chunks")
    chunk_dir.mkdir(parents=True, exist_ok=True)
    ranges = []
    offset = start
    while offset < expected_size:
        end = min(offset + chunk_size - 1, expected_size - 1)
        chunk_path = chunk_dir / f"{offset}-{end}.part"
        if not (chunk_path.exists() and chunk_path.stat().st_size == end - offset + 1):
            ranges.append((offset, end))
        offset = end + 1
    completed = 0

    with ThreadPoolExecutor(max_workers=jobs) as executor:
        futures = {
            executor.submit(_download_range_to_file, url, chunk_dir / f"{begin}-{end}.part", begin, end): (begin, end)
            for begin, end in ranges
        }
        for future in as_completed(futures):
            begin, end = futures[future]
            future.result()
            completed += end - begin + 1
            print(f"  {display_name}: fetched {start + completed}/{expected_size} bytes")

    append_ranges = [(offset, min(offset + chunk_size - 1, expected_size - 1)) for offset in range(start, expected_size, chunk_size)]
    with tmp_path.open("ab") as out:
        for begin, end in append_ranges:
            chunk_path = chunk_dir / f"{begin}-{end}.part"
            expected = end - begin + 1
            if chunk_path.stat().st_size != expected:
                raise RuntimeError(f"Chunk size mismatch for {chunk_path}: got {chunk_path.stat().st_size}, expected {expected}")
            with chunk_path.open("rb") as chunk:
                shutil.copyfileobj(chunk, out, length=1024 * 1024)
            chunk_path.unlink()

    if tmp_path.stat().st_size != expected_size:
        raise RuntimeError(f"Downloaded size mismatch for {tmp_path}: got {tmp_path.stat().st_size}, expected {expected_size}")
    chunk_dir.rmdir()


def _download_range_to_file(url: str, chunk_path: Path, start: int, end: int) -> None:
    expected = end - start + 1
    if chunk_path.exists() and chunk_path.stat().st_size == expected:
        return

    tmp_chunk = chunk_path.with_suffix(chunk_path.suffix + ".tmp")
    last_error = None
    for attempt in range(1, 8):
        done = tmp_chunk.stat().st_size if tmp_chunk.exists() else 0
        if done > expected:
            tmp_chunk.unlink()
            done = 0
        if done >= expected:
            tmp_chunk.replace(chunk_path)
            return
        range_start = start + done
        cmd = [
            "curl",
            "--fail",
            "--silent",
            "--show-error",
            "--http1.1",
            "-L",
            "--range",
            f"{range_start}-{end}",
            "--connect-timeout",
            "30",
            "--max-time",
            "300",
            "-A",
            "rxngraphormer-reproduce/1.0",
            "-o",
            "-",
            url,
        ]
        try:
            with tmp_chunk.open("ab") as out:
                subprocess.run(cmd, stdout=out, check=True)
            size = tmp_chunk.stat().st_size
            if size < expected:
                raise RuntimeError(f"Short range read for {url}: got {size}, expected {expected}")
            if size > expected:
                raise RuntimeError(f"Range over-read for {url}: got {size}, expected {expected}")
            tmp_chunk.replace(chunk_path)
            return
        except (subprocess.CalledProcessError, OSError, RuntimeError) as exc:
            last_error = exc
            sleep_s = min(2 ** attempt, 30)
            print(f"retry curl chunk {attempt}/7: {url} offset {start} failed with {exc}; sleeping {sleep_s}s")
            time.sleep(sleep_s)
    raise last_error


def _download_range(url: str, tmp_path: Path, start: int, end: int | None) -> int:
    if shutil.which("curl") and end is not None:
        last_error = None
        for attempt in range(1, 8):
            before = tmp_path.stat().st_size if tmp_path.exists() else 0
            cmd = [
                "curl",
                "--fail",
                "--silent",
                "--show-error",
                "--http1.1",
                "-L",
                "--range",
                f"{start}-{end}",
                "--retry",
                "3",
                "--connect-timeout",
                "30",
                "--max-time",
                "120",
                "-A",
                "rxngraphormer-reproduce/1.0",
                "-o",
                str(tmp_path) if start == 0 else "-",
                url,
            ]
            try:
                if start == 0:
                    subprocess.run(cmd, check=True)
                else:
                    with tmp_path.open("ab") as out:
                        cmd[-2] = "-"
                        subprocess.run(cmd, stdout=out, check=True)
                after = tmp_path.stat().st_size
                return after - before
            except subprocess.CalledProcessError as exc:
                last_error = exc
                sleep_s = min(2 ** attempt, 30)
                print(f"retry curl chunk {attempt}/7: {url} offset {start} failed; sleeping {sleep_s}s")
                time.sleep(sleep_s)
        raise last_error

    range_start = start
    headers_start = start
    if end is not None:
        range_header = f"bytes={start}-{end}"
    elif start:
        range_header = f"bytes={start}-"
    else:
        range_header = None

    last_error = None
    for attempt in range(1, 8):
        try:
            request = request_for(url, start=0)
            if range_header is not None:
                request.add_header("Range", range_header)
            with urllib.request.urlopen(request, timeout=300) as response:
                status = getattr(response, "status", 200)
                if headers_start and status != 206:
                    raise RuntimeError(f"Server did not honor Range request for {url}")
                if not headers_start and range_header and status not in (200, 206):
                    raise RuntimeError(f"Unexpected HTTP status {status} for {url}")
                mode = "r+b" if tmp_path.exists() else "wb"
                with tmp_path.open(mode) as out:
                    out.seek(range_start)
                    total = 0
                    while True:
                        block = response.read(1024 * 1024)
                        if not block:
                            break
                        out.write(block)
                        total += len(block)
                expected = None if end is None else end - range_start + 1
                if expected is not None and total != expected:
                    raise RuntimeError(f"Short read for {url}: got {total}, expected {expected}")
                return total
        except (OSError, http.client.HTTPException, RuntimeError) as exc:
            last_error = exc
            sleep_s = min(2 ** attempt, 30)
            print(f"retry chunk {attempt}/7: {url} offset {range_start} failed with {exc}; sleeping {sleep_s}s")
            time.sleep(sleep_s)
    raise last_error


def extract_zip(path: Path, destination: Path) -> None:
    with zipfile.ZipFile(path) as archive:
        archive.extractall(destination)


def select_files(args: argparse.Namespace, manifest: dict[str, dict]) -> list[str]:
    selected: set[str] = set()
    for profile in args.profile:
        selected.update(PROFILE_FILES[profile])
    selected.update(args.file)

    missing = sorted(name for name in selected if name not in manifest)
    if missing:
        raise SystemExit(f"Assets not present in figshare article {ARTICLE_ID}: {missing}")
    return sorted(selected)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument(
        "--profile",
        action="append",
        choices=sorted(PROFILE_FILES),
        default=[],
        help="Asset group to download.",
    )
    parser.add_argument("--file", action="append", default=[], help="Specific figshare file name.")
    parser.add_argument("--list", action="store_true", help="List figshare files and exit.")
    parser.add_argument("--extract", action="store_true", help="Extract downloaded zip files.")
    parser.add_argument("--jobs", type=int, default=8, help="Concurrent range requests per file.")
    parser.add_argument("--chunk-mib", type=int, default=64, help="Range chunk size in MiB.")
    parser.add_argument("--skip-existing", action="store_true", default=True)
    args = parser.parse_args()

    manifest = fetch_manifest()
    if args.list:
        for name, item in sorted(manifest.items()):
            print(f"{name}\t{item['size']}\t{item['download_url']}")
        return

    if not args.profile and not args.file:
        raise SystemExit("Choose at least one --profile or --file, or use --list.")

    for filename in select_files(args, manifest):
        item = manifest[filename]
        destination = destination_for(filename, args.root)
        zip_path = destination / filename
        if zip_path.exists() and args.skip_existing:
            print(f"exists: {zip_path}")
        else:
            print(f"download: {filename} -> {zip_path}")
            download_file(
                item["download_url"],
                zip_path,
                expected_size=item.get("size"),
                chunk_size=args.chunk_mib * 1024 * 1024,
                jobs=args.jobs,
            )
        if args.extract:
            print(f"extract: {zip_path} -> {destination}")
            extract_zip(zip_path, destination)


if __name__ == "__main__":
    main()
