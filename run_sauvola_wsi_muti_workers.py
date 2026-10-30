"""Multiprocess sauvola WSI runner with local processing functions."""

import argparse
import json
from pathlib import Path
from collections import deque
from concurrent.futures import ProcessPoolExecutor
from functools import partial
import multiprocessing
import operator
from threadpoolctl import threadpool_limits

import h5py
import numpy as np
import tqdm

from .dataset import OrionDataset, CyCIFDataset, BMCyCIFDataset, CometDataset, BaseDataset
from .utils import read_multichannel_tiles, save_tile_outputs, select_samples_to_process
from .baseline.sauvola import sauvola_2d

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


def _load_params(params=None):
    """Load the adjacent sauvola_params.json unless a dictionary is supplied."""
    if params is not None:
        return dict(params)
    with Path(__file__).with_name("sauvola_params.json").open() as file:
        return json.load(file)


def process_tile(tiles, sauvola_params=None):
    """Return (result, effective removed background, raw) for [N_c,H,W]."""
    raw = np.asarray(tiles).copy()
    params = _load_params(sauvola_params)
    if params.get("return_mask"):
        raise ValueError("Run scripts return images; return_mask must be False")
    if params.get("r") is None and np.issubdtype(raw.dtype, np.floating):
        params["r"] = 127.5
    result = np.stack([sauvola_2d(tile, **params) for tile in raw])
    return result, raw.astype(np.float64) - result.astype(np.float64), raw


def _process_coord(coord, slide_path_dict, tile_size, sauvola_params=None, save_dir=None):
    tiles = read_multichannel_tiles(slide_path_dict, coord, tile_size)
    result, bg, raw = process_tile(tiles, sauvola_params)
    if save_dir is not None:
        save_tile_outputs(save_dir, coord, list(slide_path_dict), result, bg, raw)
        return tuple(map(int, coord))
    return result, bg, raw


def process_wsi(slide_path_dict, coords, tile_size, sauvola_params=None, worker=4):
    """Yield (result,bg,raw) in coordinate order; tile_size=(H,W)."""
    sauvola_params = _load_params(sauvola_params)
    function = partial(
        _process_coord, slide_path_dict=slide_path_dict, tile_size=tile_size,
        sauvola_params=sauvola_params,
    )
    yield from tqdm.tqdm(_ordered_tile_map(function, coords, worker), total=len(coords))


def process_single_sample(dataset, sample_id, tile_size, save_dir, sauvola_params=None, worker=4):
    """Write result/bg/raw and result_raw_combinded/x_y.png for each tile."""
    sauvola_params = _load_params(sauvola_params)
    paths = dataset.get_single_multichannel(sample_id)
    chosen = {name: paths[name] for name in dataset.DEFAULT_CHANNELS}
    with h5py.File(dataset.get_coord_path(sample_id, tile_size), "r") as file:
        coords = np.asarray(file["coords"])
    if coords.ndim != 2 or coords.shape[1] != 2:
        raise ValueError("coords must have shape (N, 2)")
    if not np.issubdtype(coords.dtype, np.integer):
        raise ValueError("coords must contain integer coordinates")
    output = Path(save_dir) / sample_id
    function = partial(
        _process_coord, slide_path_dict=chosen, tile_size=(tile_size, tile_size),
        sauvola_params=sauvola_params, save_dir=output,
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
    parser.add_argument("--sauvola-params", type=Path, default=Path(__file__).with_name("sauvola_params.json"), help="Method parameter JSON")
    parser.add_argument("--worker", "--workers", type=int, default=4)
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
    if args.sauvola_params is not None:
        with args.sauvola_params.open() as file:
            params = json.load(file)
    for sample_id in sample_ids:
        process_single_sample(
            dataset, sample_id, args.tile_size, args.save_dir, sauvola_params=params, worker=args.worker,
        )


if __name__ == "__main__":
    main()
