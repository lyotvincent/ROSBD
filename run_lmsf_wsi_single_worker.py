"""Single-process lmsf WSI runner with local processing functions."""

import argparse
import json
from pathlib import Path

import h5py
import numpy as np
import tqdm

from .dataset import OrionDataset, CyCIFDataset, BMCyCIFDataset, CometDataset, BaseDataset
from .utils import read_multichannel_tiles, save_tile_outputs
from .baseline.lmsf import lmsf_2d_cumulative

def _load_params(params=None):
    """Load the adjacent lmsf_params.json unless a dictionary is supplied."""
    if params is not None:
        return dict(params)
    with Path(__file__).with_name("lmsf_params.json").open() as file:
        return json.load(file)


def process_tile(tiles, lmsf_params=None):
    """Return (result, effective removed background, raw) for [N_c,H,W]."""
    raw = np.asarray(tiles).copy()
    params = _load_params(lmsf_params)
    result = np.stack([lmsf_2d_cumulative(tile, **params) for tile in raw])
    return result, raw.astype(np.float64) - result.astype(np.float64), raw


def process_wsi(slide_path_dict, coords, tile_size, lmsf_params=None):
    """Yield (result,bg,raw) in coordinate order; tile_size=(H,W)."""
    lmsf_params = _load_params(lmsf_params)
    for coord in tqdm.tqdm(coords, total=len(coords)):
        tiles = read_multichannel_tiles(slide_path_dict, coord, tile_size)
        yield process_tile(tiles, lmsf_params)


def process_single_sample(dataset, sample_id, tile_size, save_dir, lmsf_params=None):
    """Write result/bg/raw and result_raw_combinded/x_y.png for each tile."""
    lmsf_params = _load_params(lmsf_params)
    paths = dataset.get_single_multichannel(sample_id)
    chosen = {name: paths[name] for name in dataset.DEFAULT_CHANNELS}
    with h5py.File(dataset.get_coord_path(sample_id, tile_size), "r") as file:
        coords = np.asarray(file["coords"])
    if coords.ndim != 2 or coords.shape[1] != 2:
        raise ValueError("coords must have shape (N, 2)")
    if not np.issubdtype(coords.dtype, np.integer):
        raise ValueError("coords must contain integer coordinates")
    output = Path(save_dir) / sample_id
    outputs = process_wsi(
        chosen, coords, (tile_size, tile_size), lmsf_params=lmsf_params,
    )
    for index, (result, bg, raw) in enumerate(outputs):
        save_tile_outputs(output, coords[index], list(chosen), result, bg, raw)
    return output


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", choices=["single", "cohort"], default="single")
    parser.add_argument("--dataset", choices=["orion", "cycif", "bmcycif", "comet"], default="orion")
    parser.add_argument("--sample-id", nargs="+", default=[])
    parser.add_argument("--tile-size", type=int, default=None,
                        help="Tile size: 512 for comet, 256 for other datasets by default")
    parser.add_argument("--save-dir", required=True)
    parser.add_argument("--lmsf-params", type=Path, default=Path(__file__).with_name("lmsf_params.json"), help="Method parameter JSON")
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
    params = None
    if args.lmsf_params is not None:
        with args.lmsf_params.open() as file:
            params = json.load(file)
    for sample_id in args.sample_id or dataset.sample_ids:
        process_single_sample(
            dataset, sample_id, args.tile_size, args.save_dir, lmsf_params=params,
        )


if __name__ == "__main__":
    main()
