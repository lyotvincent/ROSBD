#!/usr/bin/env python3
"""Split pyramidal Orion OME-TIFF images into single-channel OME-TIFF files."""

from __future__ import annotations

import argparse
import concurrent.futures
import os
import re
import sys
import time
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import tifffile
import zarr


TILE_SHAPE = (512, 512)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Split each Orion OME-TIFF into pyramidal single-channel "
            "OME-TIFF files."
        )
    )
    parser.add_argument("--mif-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--n-threads", "--n_threads", type=int, default=4)
    parser.add_argument(
        "--overwrite", action="store_true", help="Replace existing channel files."
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate inputs and print the planned work without writing images.",
    )
    return parser.parse_args()


def find_ome_tiffs(directory: Path) -> list[Path]:
    if not directory.is_dir():
        raise NotADirectoryError(f"mIF_dir is not a directory: {directory}")
    files = {
        path
        for pattern in ("*.ome.tif", "*.ome.tiff")
        for path in directory.glob(pattern)
        if path.is_file()
    }
    if not files:
        raise FileNotFoundError(f"No *.ome.tif or *.ome.tiff files in {directory}")
    return sorted(files)


def sample_name(path: Path) -> str:
    return re.sub(r"\.ome\.tiff?$", "", path.name, flags=re.IGNORECASE)


def safe_name(value: str) -> str:
    result = re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("._")
    if not result:
        raise ValueError(f"Channel name cannot be converted to a filename: {value!r}")
    return result


def channel_names_from_ome(ome_xml: str | None, path: Path) -> list[str]:
    if not ome_xml:
        raise ValueError(f"{path}: missing OME metadata")
    root = ET.fromstring(ome_xml)
    names = [channel.get("Name", "").strip() for channel in root.findall(".//{*}Channel")]
    if not names or any(not name for name in names):
        raise ValueError(f"{path}: one or more OME channel names are missing")
    if len(set(names)) != len(names):
        raise ValueError(f"{path}: duplicate OME channel names")
    return names


def inspect_source(path: Path) -> dict[str, object]:
    with tifffile.TiffFile(path) as tif:
        if len(tif.series) != 1:
            raise ValueError(f"{path}: expected one image series, found {len(tif.series)}")
        series = tif.series[0]
        if series.axes not in ("CYX", "IYX"):
            raise ValueError(
                f"{path}: expected axes CYX or IYX, found {series.axes} {series.shape}"
            )
        channel_count = series.shape[0]
        channel_names = channel_names_from_ome(tif.ome_metadata, path)
        level_shapes = [tuple(level.shape) for level in series.levels]

    if channel_count != len(channel_names):
        raise ValueError(
            f"{path}: image has {channel_count} planes but OME metadata has "
            f"{len(channel_names)} channel names"
        )

    return {
        "path": path,
        "channel_count": channel_count,
        "channel_names": channel_names,
        "level_shapes": level_shapes,
    }


def physical_size_metadata(ome_xml: str | None) -> dict[str, object]:
    if not ome_xml:
        return {}
    root = ET.fromstring(ome_xml)
    pixels = root.find(".//{*}Pixels")
    if pixels is None:
        return {}

    metadata: dict[str, object] = {}
    for axis in ("X", "Y"):
        value = pixels.get(f"PhysicalSize{axis}")
        unit = pixels.get(f"PhysicalSize{axis}Unit")
        if value is not None:
            metadata[f"PhysicalSize{axis}"] = float(value)
        if unit is not None:
            metadata[f"PhysicalSize{axis}Unit"] = unit
    return metadata


def tile_iterator(
    array: zarr.Array,
    channel_index: int,
    height: int,
    width: int,
):
    tile_height, tile_width = TILE_SHAPE
    for y0 in range(0, height, tile_height):
        y1 = min(y0 + tile_height, height)
        for x0 in range(0, width, tile_width):
            x1 = min(x0 + tile_width, width)
            source = np.asarray(
                array[channel_index, y0:y1, x0:x1], order="C"
            )
            if source.shape == TILE_SHAPE:
                yield source
            else:
                tile = np.zeros(TILE_SHAPE, dtype=array.dtype)
                tile[: y1 - y0, : x1 - x0] = source
                yield tile


def write_level(
    writer: tifffile.TiffWriter,
    level: tifffile.TiffPageSeries,
    channel_index: int,
    *,
    subifds: int | None = None,
    subfiletype: int | None = None,
    metadata: dict[str, object] | None = None,
) -> None:
    if level.axes not in ("CYX", "IYX"):
        raise ValueError(f"Expected level axes CYX or IYX, found {level.axes} {level.shape}")

    _, height, width = level.shape
    store = level.aszarr(maxworkers=1)
    try:
        opened = zarr.open(store, mode="r")
        # tifffile exposes the base level of a pyramid as a Zarr group whose
        # "0" member is the full-resolution array. Reduced levels are arrays.
        array = opened["0"] if isinstance(opened, zarr.Group) else opened
        writer.write(
            data=tile_iterator(array, channel_index, height, width),
            shape=(height, width),
            dtype=level.dtype,
            photometric="minisblack",
            tile=TILE_SHAPE,
            compression="deflate",
            compressionargs={"level": 6},
            predictor=True,
            subifds=subifds,
            subfiletype=subfiletype,
            metadata=metadata,
        )
    finally:
        store.close()


def output_path(output_dir: Path, source: Path, index: int, name: str) -> Path:
    directory = output_dir / sample_name(source)
    return directory / f"C{index + 1:02d}_{safe_name(name)}.ome.tif"


def write_channel(
    tif: tifffile.TiffFile,
    source: Path,
    destination: Path,
    channel_index: int,
    channel_name: str,
    overwrite: bool,
) -> str:
    if destination.is_file() and destination.stat().st_size > 0 and not overwrite:
        return "skipped"

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".partial")
    if temporary.exists():
        temporary.unlink()

    series = tif.series[0]
    metadata: dict[str, object] = {
        "axes": "YX",
        "Name": f"{sample_name(source)} - {channel_name}",
        "Channel": {"Name": [channel_name]},
    }
    metadata.update(physical_size_metadata(tif.ome_metadata))

    try:
        with tifffile.TiffWriter(temporary, bigtiff=True, ome=True) as writer:
            write_level(
                writer,
                series.levels[0],
                channel_index,
                subifds=len(series.levels) - 1,
                metadata=metadata,
            )
            for level in series.levels[1:]:
                write_level(
                    writer,
                    level,
                    channel_index,
                    subfiletype=1,
                    metadata=None,
                )
        os.replace(temporary, destination)
    except BaseException:
        if temporary.exists():
            temporary.unlink()
        raise
    return "written"


def write_manifest(
    output_dir: Path,
    source: Path,
    channel_names: list[str],
) -> None:
    directory = output_dir / sample_name(source)
    directory.mkdir(parents=True, exist_ok=True)
    destination = directory / "channels.tsv"
    temporary = directory / "channels.tsv.partial"
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        handle.write("channel_index\tchannel_number\tchannel_name\tfile\n")
        for index, name in enumerate(channel_names):
            path = output_path(output_dir, source, index, name)
            handle.write(f"{index}\t{index + 1}\t{name}\t{path.name}\n")
    os.replace(temporary, destination)


def process_source(
    source_text: str,
    output_dir_text: str,
    channel_names: list[str],
    overwrite: bool,
) -> dict[str, object]:
    source = Path(source_text)
    output_dir = Path(output_dir_text)
    started = time.monotonic()
    written = 0
    skipped = 0

    with tifffile.TiffFile(source) as tif:
        series = tif.series[0]
        if series.axes not in ("CYX", "IYX") or series.shape[0] != len(channel_names):
            raise ValueError(
                f"{source}: source changed after validation; found "
                f"{series.axes} {series.shape}, expected {len(channel_names)} channels"
            )
        for index, name in enumerate(channel_names):
            destination = output_path(output_dir, source, index, name)
            status = write_channel(
                tif, source, destination, index, name, overwrite=overwrite
            )
            written += status == "written"
            skipped += status == "skipped"

    write_manifest(output_dir, source, channel_names)
    return {
        "source": source.name,
        "written": written,
        "skipped": skipped,
        "seconds": time.monotonic() - started,
    }


def main() -> int:
    args = parse_args()
    if args.n_threads <= 0:
        raise ValueError("n_threads must be greater than zero")

    sources = find_ome_tiffs(args.mif_dir)
    plans = [inspect_source(path) for path in sources]

    print(f"Input directory : {args.mif_dir}")
    print(f"Output directory: {args.output_dir}")
    print("Channel names   : read from each file's OME metadata")
    print(f"Source images   : {len(plans)}")
    print(f"Workers         : {args.n_threads}")
    if args.dry_run:
        total = sum(int(plan["channel_count"]) for plan in plans)
        print(f"Dry run successful: would produce {total} single-channel files.")
        return 0

    args.output_dir.mkdir(parents=True, exist_ok=True)
    workers = min(args.n_threads, len(plans))
    failures: list[tuple[str, BaseException]] = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(
                process_source,
                str(plan["path"]),
                str(args.output_dir),
                list(plan["channel_names"]),
                args.overwrite,
            ): Path(plan["path"]).name
            for plan in plans
        }
        completed = 0
        for future in concurrent.futures.as_completed(futures):
            source_name = futures[future]
            try:
                result = future.result()
            except BaseException as exc:
                failures.append((source_name, exc))
                print(f"ERROR: {source_name}: {exc}", file=sys.stderr, flush=True)
            else:
                completed += 1
                print(
                    f"[{completed}/{len(plans)}] {result['source']}: "
                    f"written={result['written']}, skipped={result['skipped']}, "
                    f"elapsed={result['seconds']:.1f}s",
                    flush=True,
                )

    if failures:
        print(f"Failed images: {len(failures)}", file=sys.stderr)
        for source_name, exc in failures:
            print(f"  {source_name}: {exc}", file=sys.stderr)
        return 1

    print("All Orion images completed successfully.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
