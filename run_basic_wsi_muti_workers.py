"""Multiprocess basic WSI runner with local processing functions."""

import argparse
import json
import operator
from pathlib import Path
from collections import deque
from concurrent.futures import ProcessPoolExecutor
from functools import partial
import multiprocessing
from threadpoolctl import threadpool_limits

import h5py
import numpy as np
import tqdm

from .dataset import OrionDataset, CyCIFDataset, BMCyCIFDataset, CometDataset, BaseDataset
from .utils import read_multichannel_tiles, save_tile_outputs, select_samples_to_process

_THREAD_LIMIT = None


def _initialize_worker():
    global _THREAD_LIMIT
    _THREAD_LIMIT = threadpool_limits(limits=1)


def _ordered_tile_map(function, coords, worker):
    """Yield in order with at most 2*worker pending jobs."""
    if isinstance(worker, bool):
        raise ValueError("worker must be a positive integer")
    try:
        worker = operator.index(worker)
    except TypeError:
        raise ValueError("worker must be a positive integer") from None
    if worker < 1:
        raise ValueError("worker must be a positive integer")
    if len(coords) == 0:
        return
    if worker == 1:
        for coord in coords:
            yield function(coord)
        return
    worker = min(worker, len(coords))
    iterator = iter(coords)
    with ProcessPoolExecutor(
        max_workers=worker,
        mp_context=multiprocessing.get_context("spawn"),
        initializer=_initialize_worker,
    ) as pool:
        pending = deque()
        try:
            for _ in range(min(2*worker, len(coords))):
                pending.append(pool.submit(function, next(iterator)))
            while pending:
                result = pending.popleft().result()
                yield result
                try:
                    coord = next(iterator)
                except StopIteration:
                    continue
                pending.append(pool.submit(function, coord))
        finally:
            for future in pending:
                future.cancel()


def fit_basic_profiles(slide_path_dict, coords, tile_size, basic_params=None, fit_samples=32):
    """Fit each channel independently on evenly spaced coordinates.

    Fit at most fit_samples tiles per channel, bounding training memory.
    Return only NumPy flatfield/darkfield arrays; no model/GPU state is sent
    to workers. Reuse these profiles to compare worker counts identically.
    """
    from .baseline.basic import fit_basic

    if isinstance(fit_samples, bool):
        raise ValueError("fit_samples must be an integer >= 2")
    try:
        fit_samples = operator.index(fit_samples)
    except TypeError:
        raise ValueError("fit_samples must be an integer >= 2") from None
    if fit_samples < 2 or len(coords) < 2:
        raise ValueError("BaSiC fitting requires at least two sample tiles")
    indices = np.linspace(0, len(coords) - 1, min(fit_samples, len(coords)), dtype=int)
    profiles = {}
    for name, path in slide_path_dict.items():
        images = np.stack([
            read_multichannel_tiles({name: path}, coords[index], tile_size)[0]
            for index in indices
        ])
        model = fit_basic(images, **dict(basic_params or {}))
        profiles[name] = {
            "flatfield": np.array(model.flatfield, copy=True),
            "darkfield": np.array(model.darkfield, copy=True),
        }
    return profiles


def fit_bm_basic_profiles(dataset, basic_params=None, fit_samples=32):
    """Fit one illumination field per physical channel from complete BM views.

    All non-excluded scenes are candidates, even for a single-sample output.
    Views are resized onto a common normalized-coordinate grid, since the
    registered TIFF dimensions differ. Profiles stay small for worker IPC.
    fit_samples limits views per channel here, rather than patches.
    """
    import tifffile
    import cv2
    from .baseline.basic import fit_basic

    fit_samples = operator.index(fit_samples)
    if fit_samples < 2 or len(dataset.sample_ids) < 2:
        raise ValueError("BM BaSiC fitting requires at least two complete views")
    indices = np.linspace(
        0, len(dataset.sample_ids) - 1,
        min(fit_samples, len(dataset.sample_ids)), dtype=int,
    )
    sample_ids = [dataset.sample_ids[i] for i in indices]
    params = dict(basic_params or {})
    params.setdefault("get_darkfield", False)
    grid_size = params.get("working_size", 128)
    if grid_size is None:
        grid_size = 128
    grid_size = operator.index(grid_size)
    if grid_size < 2:
        raise ValueError("BM BaSiC working_size must be at least 2")
    profiles = {}
    for name in tqdm.tqdm(dataset.DEFAULT_CHANNELS, desc="BaSiC full-view fitting"):
        images = []
        for sample_id in sample_ids:
            path = dataset.get_single_multichannel(sample_id)[name]
            with tifffile.TiffFile(path) as file:
                image = file.pages[0].asarray()
            if image.ndim != 2:
                raise ValueError(f"Expected a 2-D BM view: {path}")
            images.append(cv2.resize(
                image.astype(np.float32), (grid_size, grid_size),
                interpolation=cv2.INTER_AREA,
            ))
        model = fit_basic(np.stack(images), **params)
        profiles[name] = {
            "flatfield": np.array(model.flatfield, copy=True),
            "darkfield": np.array(model.darkfield, copy=True),
            "space": "full_view",
            "min_flatfield": 0.01,
            "invalid_flatfield_value": 1.0,
        }
    return profiles


def _bm_sample_profiles(dataset, sample_id, profiles):
    """Attach each source image's dimensions to its shared illumination field."""
    import tifffile

    paths = dataset.get_single_multichannel(sample_id)
    selected = {}
    for name in dataset.DEFAULT_CHANNELS:
        profile = profiles[name]
        if profile.get("space") != "full_view":
            raise ValueError("BM requires full-view BaSiC profiles")
        with tifffile.TiffFile(paths[name]) as file:
            shape = file.pages[0].shape
        selected[name] = {**profile, "image_shape": shape}
    return selected


def _profile_tile(profile, coord, tile_shape):
    """Sample the full-view field; BM flatfield <= 0.01/nonfinite falls back to 1."""
    from scipy.ndimage import map_coordinates

    if profile.get("space") != "full_view":
        return profile["flatfield"], profile["darkfield"]
    if coord is None:
        raise ValueError("Full-view BaSiC profiles require the tile (x, y)")
    height, width = profile["image_shape"]
    y = np.arange(tile_shape[0], dtype=np.float64) + int(coord[1])
    x = np.arange(tile_shape[1], dtype=np.float64) + int(coord[0])
    fields = []
    valid = (y[:, None] >= 0) & (y[:, None] < height)
    valid = valid & (x[None, :] >= 0) & (x[None, :] < width)
    for key, outside in (("flatfield", 1.0), ("darkfield", 0.0)):
        field = np.asarray(profile[key])
        yy = (y + 0.5) * field.shape[0] / height - 0.5
        xx = (x + 0.5) * field.shape[1] / width - 0.5
        grid = np.meshgrid(yy, xx, indexing="ij")
        values = map_coordinates(field, grid, order=1, mode="nearest")
        values = np.where(valid, values, outside)
        if key == "flatfield":
            # BM full-view fields are mean-normalized to 1. Replace invalid
            # or near-zero values after interpolation, avoiding zero crossings.
            minimum = float(profile.get("min_flatfield", 0.01))
            fallback = float(profile.get("invalid_flatfield_value", 1.0))
            if not np.isfinite(minimum) or minimum < 0:
                raise ValueError("min_flatfield must be finite and nonnegative")
            if not np.isfinite(fallback) or fallback <= minimum:
                raise ValueError("invalid_flatfield_value must exceed min_flatfield")
            values = np.where(np.isfinite(values) & (values > minimum), values, fallback)
        fields.append(values)
    return tuple(fields)


def _prepare_profiles(slide_path_dict, coords, tile_size, basic_params, profiles, fit_samples):
    if len(coords):
        if profiles is None:
            profiles = fit_basic_profiles(
                slide_path_dict, coords, tile_size, basic_params, fit_samples
            )
        if list(profiles) != list(slide_path_dict):
            raise ValueError("BaSiC profile names/order must match slide channels")
    return profiles


def process_tile(tiles, profiles, coord=None):
    """Return (result, effective removed background, raw) for [N_c,H,W]."""
    raw = np.asarray(tiles).copy()
    if profiles is None or len(profiles) != len(raw):
        raise ValueError("Provide one fitted BaSiC profile per channel")
    result = []
    for tile, profile in zip(raw, profiles.values()):
        flatfield, darkfield = _profile_tile(profile, coord, tile.shape)
        corrected = (tile.astype(np.float32) - darkfield) / flatfield
        result.append(np.maximum(corrected, 0).astype(np.float32))
    result = np.stack(result)
    return result, raw.astype(np.float64) - result.astype(np.float64), raw


def _process_coord(coord, slide_path_dict, tile_size, profiles, save_dir=None):
    tiles = read_multichannel_tiles(slide_path_dict, coord, tile_size)
    result, bg, raw = process_tile(tiles, profiles, coord=coord)
    if save_dir is not None:
        save_tile_outputs(save_dir, coord, list(slide_path_dict), result, bg, raw)
        return tuple(map(int, coord))
    return result, bg, raw


def process_wsi(slide_path_dict, coords, tile_size, basic_params=None, worker=4, profiles=None, fit_samples=32):
    """Yield (result,bg,raw) in coordinate order; tile_size=(H,W)."""
    profiles = _prepare_profiles(
        slide_path_dict, coords, tile_size, basic_params, profiles, fit_samples
    )
    function = partial(
        _process_coord, slide_path_dict=slide_path_dict, tile_size=tile_size,
        profiles=profiles,
    )
    yield from tqdm.tqdm(_ordered_tile_map(function, coords, worker), total=len(coords))


def process_single_sample(dataset, sample_id, tile_size, save_dir, basic_params=None, worker=4, profiles=None, fit_samples=32):
    """Write result/bg/raw and result_raw_combinded/x_y.png for each tile."""
    if isinstance(dataset, BMCyCIFDataset):
        if profiles is None:
            profiles = fit_bm_basic_profiles(dataset, basic_params, fit_samples)
        profiles = _bm_sample_profiles(dataset, sample_id, profiles)
    paths = dataset.get_single_multichannel(sample_id)
    chosen = {name: paths[name] for name in dataset.DEFAULT_CHANNELS}
    with h5py.File(dataset.get_coord_path(sample_id, tile_size), "r") as file:
        coords = np.asarray(file["coords"])
    if coords.ndim != 2 or coords.shape[1] != 2:
        raise ValueError("coords must have shape (N, 2)")
    if not np.issubdtype(coords.dtype, np.integer):
        raise ValueError("coords must contain integer coordinates")
    output = Path(save_dir) / sample_id
    profiles = _prepare_profiles(
        chosen, coords, (tile_size, tile_size), basic_params, profiles, fit_samples
    )
    function = partial(
        _process_coord, slide_path_dict=chosen, tile_size=(tile_size, tile_size),
        profiles=profiles, save_dir=output,
    )
    for _ in tqdm.tqdm(_ordered_tile_map(function, coords, worker), total=len(coords)):
        pass
    return output


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", choices=["single", "cohort"], default="single")
    parser.add_argument("--dataset", choices=["orion", "cycif", "bmcycif", "comet"], default="orion")
    parser.add_argument("--sample-id", nargs="+", default=[])
    parser.add_argument("--tile-size", type=int, default=None,
                        help="Tile size: 512 for comet, 256 for other datasets by default")
    parser.add_argument("--save-dir", required=True)
    parser.add_argument("--overwrite", action="store_true",
                        help="Reprocess samples with existing output directories")
    parser.add_argument("--basic-params", type=Path, help="Optional method parameter JSON")
    parser.add_argument("--worker", "--workers", type=int, default=4)
    parser.add_argument("--fit-samples", type=int, default=32,
                        help="Max fitting tiles; for bmcycif, max complete views")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.tile_size is None:
        args.tile_size = 512 if args.dataset == "comet" else 256
    if args.task == "single" and not args.sample_id:
        raise ValueError("--sample-id is required for --task single")
    dataset = {
        "orion": OrionDataset, "cycif": CyCIFDataset,
        "bmcycif": BMCyCIFDataset, "comet": CometDataset,
    }[args.dataset]()
    sample_ids = select_samples_to_process(
        args.sample_id or dataset.sample_ids, args.save_dir, args.overwrite,
    )
    if not sample_ids:
        return
    params = None
    if args.basic_params is not None:
        with args.basic_params.open() as file:
            params = json.load(file)
    profiles = None
    if isinstance(dataset, BMCyCIFDataset):
        profiles = fit_bm_basic_profiles(dataset, params, args.fit_samples)
    for sample_id in sample_ids:
        process_single_sample(
            dataset, sample_id, args.tile_size, args.save_dir, basic_params=params, worker=args.worker, profiles=profiles, fit_samples=args.fit_samples,
        )


if __name__ == "__main__":
    main()
