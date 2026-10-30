#!/usr/bin/env python3
"""Generate patch coordinates from DAPI/Hoechst or BM-CyCIF reference TIFFs."""

from __future__ import annotations

import argparse
import multiprocessing as mp
import os
import re
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2
import h5py
import numpy as np
import tifffile
from PIL import Image
from tqdm import tqdm

from utils.patching_wsi import get_patch_tissue_ratio

# Match both Orion Hoechst naming variants and only the first CyCIF round.
DEFAULT_DAPI_REGEX = r"(?:dapi|hoechst(?:1)?(?:\D|$)|nucleus[_ -]?hoechst)"
BM_FILE_REGEX = re.compile(
    r"Registered-R(?P<round>\d+)_[^_]+_"
    r"(?P<sample>.+-Scene-\d+)_c(?P<channel>\d+)_ORG\.tiff?",
    flags=re.IGNORECASE,
)
_PATCH_CONTEXT = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create HDF5 patch coordinates from single-channel DAPI/Hoechst "
            "TIFFs or a BM-Her2N75-15 reference channel."
        )
    )
    parser.add_argument(
        "--mif-dir",
        "-d",
        default="/data/wzx/BM-Her2N75-15",
        type=Path,
    )
    parser.add_argument(
        "--save-dir",
        "-s",
        default=None,
        type=Path,
        help="Output directory; default: <mif-dir>/patch_info/<size>_<size>.",
    )
    parser.add_argument("--patch-size", "-ps", type=int, default=512)
    parser.add_argument("--overlap", "-ol", type=int, default=0)
    parser.add_argument("--n-threads", "--n_threads", type=int, default=64)
    parser.add_argument("--chunk-size", type=int, default=512)
    parser.add_argument("--target-size", type=int, default=6000)
    parser.add_argument(
        "--min-dapi-ratio",
        type=float,
        default=0.01,
        help="Minimum fraction of DAPI-positive pixels required in a patch.",
    )
    parser.add_argument(
        "--dapi-threshold",
        type=float,
        default=None,
        help=(
            "Absolute DAPI intensity threshold. By default a log-scaled Otsu "
            "threshold is estimated independently for each image."
        ),
    )
    parser.add_argument(
        "--dapi-regex",
        default=None,
        help=(
            "Custom case-insensitive reference filename regex. If omitted, "
            "BM files use --reference-round/channel; other files use DAPI/Hoechst."
        ),
    )
    parser.add_argument(
        "--sample-from",
        choices=("auto", "parent", "filename", "scene"),
        default="auto",
        help="auto extracts BM scene IDs, otherwise uses the parent directory.",
    )
    parser.add_argument("--reference-round", type=int, default=1,
                        help="BM reference round (default R1); ignored for a custom regex.")
    parser.add_argument("--reference-channel", type=int, default=1,
                        help="BM reference channel (default c1); not an inferred marker name.")
    parser.add_argument("--stitch", "-st", action="store_true")
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if not args.mif_dir.is_dir():
        raise NotADirectoryError(f"mIF directory not found: {args.mif_dir}")
    if args.patch_size <= 0:
        raise ValueError("patch_size must be > 0")
    if args.overlap < 0 or args.overlap >= args.patch_size:
        raise ValueError("overlap must satisfy 0 <= overlap < patch_size")
    if args.n_threads <= 0:
        raise ValueError("n_threads must be > 0")
    if args.chunk_size <= 0:
        raise ValueError("chunk_size must be > 0")
    if args.target_size <= 0:
        raise ValueError("target_size must be > 0")
    if not 0.0 <= args.min_dapi_ratio <= 1.0:
        raise ValueError("min_dapi_ratio must be between 0 and 1")
    if args.dapi_threshold is not None and args.dapi_threshold < 0:
        raise ValueError("dapi_threshold must be >= 0")
    if args.reference_round <= 0 or args.reference_channel <= 0:
        raise ValueError("reference_round and reference_channel must be > 0")


def strip_tiff_suffix(name: str) -> str:
    return re.sub(r"(?:\.ome)?\.tiff?$", "", name, flags=re.IGNORECASE)


def sample_id(path: Path, mode: str) -> str:
    if mode in ("auto", "scene"):
        match = BM_FILE_REGEX.fullmatch(path.name)
        if match is not None:
            return match.group("sample")
        if mode == "scene":
            raise ValueError(f"Cannot extract a BM scene ID from {path.name}")
        mode = "parent"
    if mode == "parent":
        return path.parent.name
    if mode == "filename":
        return strip_tiff_suffix(path.name)
    raise ValueError(f"Unknown sample ID mode: {mode}")


def find_dapi_images(
    root: Path, pattern: str | None, sample_mode: str,
    reference_round: int = 1, reference_channel: int = 1,
) -> list[tuple[str, Path]]:
    """Find one reference per sample, including flat BM round/channel TIFFs.

    An explicit regex overrides automatic reference selection. Channel numbers
    identify files only; no DAPI/AF marker identity is inferred for BM files.
    """
    candidates = sorted(
        path for path in root.rglob("*")
        if path.is_file() and path.suffix.lower() in (".tif", ".tiff")
    )
    bm_files = [(path, BM_FILE_REGEX.fullmatch(path.name)) for path in candidates]
    bm_files = [(path, match) for path, match in bm_files if match is not None]
    if pattern is None and (bm_files or sample_mode == "scene"):
        images = [
            path for path, match in bm_files
            if int(match.group("round")) == reference_round
            and int(match.group("channel")) == reference_channel
        ]
        description = f"BM reference R{reference_round}_c{reference_channel}"
    else:
        pattern = DEFAULT_DAPI_REGEX if pattern is None else pattern
        matcher = re.compile(pattern, flags=re.IGNORECASE)
        images = [path for path in candidates if matcher.search(path.name)]
        description = f"reference regex {pattern!r}"
    if not images:
        raise FileNotFoundError(f"No TIFF images under {root} matching {description}")

    samples = [(sample_id(path, sample_mode), path) for path in images]
    counts: dict[str, int] = {}
    for name, _ in samples:
        counts[name] = counts.get(name, 0) + 1
    duplicates = sorted(name for name, count in counts.items() if count > 1)
    if duplicates:
        preview = ", ".join(duplicates[:5])
        raise ValueError(
            f"Multiple reference images map to the same sample ID ({preview}). "
            "Narrow --dapi-regex or change --sample-from (BM: auto/scene)."
        )
    return samples


def read_mask_level(path: Path, target_size: int) -> tuple[np.ndarray, int, int, int]:
    with tifffile.TiffFile(path) as tif:
        if len(tif.series) != 1:
            raise ValueError(
                f"{path}: expected one image series, found {len(tif.series)}"
            )
        series = tif.series[0]
        if series.axes != "YX":
            raise ValueError(
                f"{path}: expected a single-channel YX image, found "
                f"{series.axes} {series.shape}"
            )
        height, width = series.shape
        levels = series.levels
        level_index = min(
            range(len(levels)),
            key=lambda index: abs(max(levels[index].shape) - target_size),
        )
        lowres = levels[level_index].asarray()

    if lowres.ndim != 2:
        raise ValueError(f"{path}: mask level must be 2-D, got {lowres.shape}")
    return lowres, width, height, level_index


def log_normalize_uint8(image: np.ndarray) -> tuple[np.ndarray, float, float]:
    sample = image[::4, ::4]
    low, high = np.percentile(sample, (1.0, 99.5))
    low = max(0.0, float(low))
    high = float(high)
    if not np.isfinite(high) or high <= low:
        high = low + 1.0

    log_low = np.log1p(low)
    log_high = np.log1p(high)
    if np.issubdtype(image.dtype, np.integer) and image.dtype.itemsize <= 2:
        values = np.arange(np.iinfo(image.dtype).max + 1, dtype=np.float32)
        values = (np.log1p(values) - log_low) * (255.0 / (log_high - log_low))
        lookup = np.clip(values, 0, 255).astype(np.uint8)
        normalized = lookup[image]
    else:
        values = image.astype(np.float32, copy=True)
        np.log1p(np.maximum(values, 0, out=values), out=values)
        values -= log_low
        values *= 255.0 / (log_high - log_low)
        normalized = np.clip(values, 0, 255).astype(np.uint8)
    return normalized, low, high


def make_dapi_mask(
    image: np.ndarray, absolute_threshold: float | None
) -> tuple[np.ndarray, np.ndarray, float]:
    normalized, low, high = log_normalize_uint8(image)
    if absolute_threshold is None:
        otsu_threshold, mask = cv2.threshold(
            normalized, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )
        log_low = np.log1p(low)
        log_high = np.log1p(high)
        threshold = float(
            np.expm1(log_low + (otsu_threshold / 255.0) * (log_high - log_low))
        )
    else:
        threshold = float(absolute_threshold)
        mask = np.where(image > threshold, 255, 0).astype(np.uint8)

    return mask, normalized, threshold


def set_patch_context(
    mask: np.ndarray,
    scale_x: float,
    scale_y: float,
    width: int,
    height: int,
    patch_size: int,
    min_dapi_ratio: float,
) -> None:
    global _PATCH_CONTEXT
    _PATCH_CONTEXT = (
        mask,
        scale_x,
        scale_y,
        width,
        height,
        patch_size,
        min_dapi_ratio,
    )


def filter_patch_chunk(coords: list[tuple[int, int]]) -> list[tuple[int, int]]:
    mask, scale_x, scale_y, width, height, patch_size, min_ratio = _PATCH_CONTEXT
    selected = []
    for x, y in coords:
        ratio = get_patch_tissue_ratio(
            x=x,
            y=y,
            patch_size=patch_size,
            tissue_mask=mask,
            scale_x=scale_x,
            scale_y=scale_y,
            wsi_width=width,
            wsi_height=height,
        )
        if ratio >= min_ratio:
            selected.append((x, y))
    return selected


def coordinate_chunks(width: int, height: int, stride: int, chunk_size: int):
    chunk = []
    for y in range(0, height, stride):
        for x in range(0, width, stride):
            chunk.append((x, y))
            if len(chunk) == chunk_size:
                yield chunk
                chunk = []
    if chunk:
        yield chunk


def save_stitch(
    path: Path,
    normalized: np.ndarray,
    coords: list[tuple[int, int]],
    patch_size: int,
    scale_x: float,
    scale_y: float,
) -> None:
    coverage = np.zeros(normalized.shape, dtype=bool)
    height, width = normalized.shape
    for x, y in coords:
        x1 = max(0, int(np.floor(x / scale_x)))
        y1 = max(0, int(np.floor(y / scale_y)))
        x2 = min(width, int(np.ceil((x + patch_size) / scale_x)))
        y2 = min(height, int(np.ceil((y + patch_size) / scale_y)))
        coverage[y1:y2, x1:x2] = True
    preview = np.where(coverage, normalized, 0).astype(np.uint8)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(preview, mode="L").save(path)


def process_image(
    sample: str,
    image_path: Path,
    save_dir: Path,
    args: argparse.Namespace,
) -> int:
    lowres, width, height, level_index = read_mask_level(image_path, args.target_size)
    mask, normalized, threshold = make_dapi_mask(lowres, args.dapi_threshold)
    scale_x = width / mask.shape[1]
    scale_y = height / mask.shape[0]

    stride = args.patch_size - args.overlap
    total_patches = ((width - 1) // stride + 1) * ((height - 1) // stride + 1)
    total_chunks = (total_patches + args.chunk_size - 1) // args.chunk_size
    chunks = coordinate_chunks(width, height, stride, args.chunk_size)

    set_patch_context(
        mask,
        scale_x,
        scale_y,
        width,
        height,
        args.patch_size,
        args.min_dapi_ratio,
    )
    cv2.setNumThreads(1)
    coords: list[tuple[int, int]] = []
    with ProcessPoolExecutor(
        max_workers=min(args.n_threads, total_chunks),
        mp_context=mp.get_context("fork"),
    ) as executor:
        results = executor.map(filter_patch_chunk, chunks)
        for selected in tqdm(
            results,
            total=total_chunks,
            desc=sample,
            unit="chunk",
            leave=False,
        ):
            coords.extend(selected)

    save_path = save_dir / f"{sample}.h5"
    with h5py.File(save_path, "w") as handle:
        handle.create_dataset(
            "coords",
            data=np.asarray(coords, dtype=np.int32).reshape(-1, 2),
            compression="gzip",
        )
        handle.attrs["sample_id"] = sample
        handle.attrs["dapi_path"] = os.fspath(image_path)
        match = BM_FILE_REGEX.fullmatch(image_path.name)
        if match is not None:
            handle.attrs["reference_round"] = int(match.group("round"))
            handle.attrs["reference_channel"] = int(match.group("channel"))
        handle.attrs["dapi_threshold"] = threshold
        handle.attrs["mask_level"] = level_index
        handle.attrs["min_dapi_ratio"] = args.min_dapi_ratio
        handle.attrs["patch_size"] = args.patch_size
        handle.attrs["overlap"] = args.overlap
        handle.attrs["width"] = width
        handle.attrs["height"] = height

    if args.stitch:
        save_stitch(
            save_dir / sample / "stitch.png",
            normalized,
            coords,
            args.patch_size,
            scale_x,
            scale_y,
        )
    return len(coords)


def main() -> None:
    args = parse_args()
    validate_args(args)
    samples = find_dapi_images(
        args.mif_dir, args.dapi_regex, args.sample_from,
        reference_round=args.reference_round, reference_channel=args.reference_channel,
    )
    if args.save_dir is None:
        args.save_dir = args.mif_dir / "patch_info" / f"{args.patch_size}_{args.patch_size}"
    args.save_dir.mkdir(parents=True, exist_ok=True)

    progress = tqdm(samples, desc="mIF images", unit="image")
    for sample, image_path in progress:
        count = process_image(sample, image_path, args.save_dir, args)
        progress.set_postfix(sample=sample, patches=count)


if __name__ == "__main__":
    main()
