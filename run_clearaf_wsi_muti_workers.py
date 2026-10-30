"""Multiprocess ClearAF tile processing.
"""

import argparse
from functools import partial
from pathlib import Path
import h5py
import numpy as np
import tqdm
from collections import deque
from concurrent.futures import ProcessPoolExecutor
import multiprocessing
import operator
from threadpoolctl import threadpool_limits

from .dataset import OrionDataset, CyCIFDataset, BMCyCIFDataset, CometDataset, BaseDataset
from .run_clearaf_wsi_single_worker import process_tile
from .utils import read_multichannel_tiles, save_tile_outputs, select_samples_to_process



_THREAD_LIMIT = None

def _initialize_worker():
    # Prevent each process from starting a full BLAS/OpenMP thread pool.
    global _THREAD_LIMIT
    _THREAD_LIMIT = threadpool_limits(limits=1)

def ordered_tile_map(function, coords, worker):
    """Yield in input order, with at most 2 * worker submitted tile jobs."""
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
            for _ in range(min(2 * worker, len(coords))):
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
    parser.add_argument("--worker", "--workers", type=int, default=4)
    return parser.parse_args()


def _process_coord(coord, slide_path_dict, tile_size, af_path, save_dir=None):
    # TIFF readers and temporary files are created inside each worker.
    tiles = read_multichannel_tiles(slide_path_dict, coord, tile_size)
    af_ref = read_multichannel_tiles({"AF": af_path}, coord, tile_size)[0]
    try:
        result, bg, raw = process_tile(tiles, af_ref)
    except Exception as error:
        raise RuntimeError(
            f"ClearAF failed at coord={tuple(map(int, coord))}, "
            f"channels={list(slide_path_dict)}, AF={af_path}"
        ) from error
    if save_dir is not None:
        save_tile_outputs(save_dir, coord, list(slide_path_dict), result, bg, raw)
        return tuple(map(int, coord))
    return result, bg, raw


def process_wsi(slide_path_dict, af_path, coords, tile_size, worker=4):
    """Yield (result, bg, raw) in coords order using worker processes.

    tile_size is (H, W); worker=1 runs sequentially. At most 2*worker
    jobs are queued, bounding the number of buffered tile results.
    """
    function = partial(
        _process_coord, slide_path_dict=slide_path_dict, tile_size=tile_size,
        af_path=af_path,
    )
    yield from tqdm.tqdm(
        ordered_tile_map(function, coords, worker), total=len(coords)
    )


def process_single_sample(dataset: BaseDataset, sample_id, tile_size, save_dir, worker=4):
    """Process and save a sample, matching the single-process output layout.

    Each process reads, corrects and saves its tile. Only coordinates are
    returned to the parent, avoiding interprocess transfer of image arrays.
    Returns save_dir/sample_id after all jobs have completed.
    """
    if dataset.af is None:
        raise ValueError("ClearAF requires a configured AF reference; bmcycif and comet are not supported")
    paths = dataset.get_single_multichannel(sample_id)
    chosen = {name: paths[name] for name in dataset.DEFAULT_CHANNELS}
    with h5py.File(dataset.get_coord_path(sample_id, tile_size), "r") as file:
        coords = np.asarray(file["coords"])
    if coords.ndim != 2 or coords.shape[1] != 2:
        raise ValueError("coords must have shape (N, 2)")
    sample_dir = Path(save_dir) / sample_id
    function = partial(
        _process_coord, slide_path_dict=chosen, tile_size=(tile_size, tile_size),
        af_path=paths[dataset.af], save_dir=sample_dir,
    )
    for _ in tqdm.tqdm(ordered_tile_map(function, coords, worker), total=len(coords)):
        pass
    return sample_dir


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
    if dataset.af is None:
        raise ValueError("ClearAF requires a configured AF reference; bmcycif and comet are not supported")
    sample_ids = select_samples_to_process(
        args.sample_id or dataset.sample_ids, args.save_dir, args.overwrite,
    )
    if not sample_ids:
        return
    for sample_id in sample_ids:
        process_single_sample(
            dataset, sample_id, args.tile_size, args.save_dir,
            worker=args.worker,
        )


if __name__ == "__main__":
    main()
