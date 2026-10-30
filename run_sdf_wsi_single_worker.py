import json
import h5py
import numpy as np
from pathlib import Path
import tqdm
import argparse

from .dataset import OrionDataset, CyCIFDataset, BMCyCIFDataset, CometDataset, BaseDataset
from .method import shared_background_filter
from .utils import read_multichannel_tiles, save_tile_outputs


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', choices=['single', 'cohort'], default='single')
    parser.add_argument('--dataset', choices=['orion', 'cycif', 'bmcycif', 'comet'], default='orion')
    parser.add_argument('--sample-id', nargs='+', default=[])
    parser.add_argument('--tile-size', type=int, default=None,
                        help='Tile size: 512 for comet, 256 for other datasets by default')
    parser.add_argument('--save-dir', required=True)
    parser.add_argument('--sdf-params', type=Path, help='Optional SDF parameter JSON')
    return parser.parse_args()


def process_tile(tiles, sdf_params):
    """
    Args:
        tiles: [N_c, H, W]
    Returns:
        result: [N_c, H, W]
    """
    raw = tiles.copy()
    if sdf_params:
        result, _, bg = shared_background_filter(tiles, **sdf_params)
    else:
        result, _, bg = shared_background_filter(tiles)
    return result, bg, raw


def process_wsi(slide_path_dict, coords, tile_size, sdf_params=None):
    for coord in tqdm.tqdm(coords, total=len(coords)):
        multichannel_tiles = read_multichannel_tiles(slide_path_dict, coord, tile_size)
        result, bg, raw = process_tile(multichannel_tiles, sdf_params)
        yield result, bg, raw
    

def process_single_sample(dataset: BaseDataset, sample_id, tile_size, save_dir, sdf_params=None):
    """Save result/bg/raw channel grids and result-left/raw-right PNGs.

    Uses the same selected Orion channels and output layout as ClearAF:
    save_dir/sample_id/{result,bg,raw,result_raw_combinded}/x_y.png.
    PNG previews clip to [0, 255] without per-image normalization.
    sdf_params is forwarded to shared_background_filter; None uses its defaults.
    Returns the sample output directory.
    """
    slide_path_dict = dataset.get_single_multichannel(sample_id)
    chosen_slide_path_dict = {name: slide_path_dict[name] for name in dataset.DEFAULT_CHANNELS}
    with h5py.File(dataset.get_coord_path(sample_id, tile_size), "r") as file:
        coords = np.asarray(file["coords"])
    sample_dir = Path(save_dir) / sample_id
    outputs = process_wsi(
        chosen_slide_path_dict, coords, (tile_size, tile_size), sdf_params=sdf_params
    )
    for index, (result, bg, raw) in enumerate(outputs):
        coord = coords[index]
        save_tile_outputs(sample_dir, coord, dataset.DEFAULT_CHANNELS, result, bg, raw)
    return sample_dir


def main():
    args = parse_args()
    if args.tile_size is None:
        args.tile_size = 512 if args.dataset == "comet" else 256
    dataset = {
        "orion": OrionDataset, "cycif": CyCIFDataset,
        "bmcycif": BMCyCIFDataset, "comet": CometDataset,
    }[args.dataset]()
    if args.task == "single" and not args.sample_id:
        raise ValueError("--sample-id is required for --task single")
    params = None
    if args.sdf_params is not None:
        with args.sdf_params.open() as file:
            params = json.load(file)
    for sample_id in args.sample_id or dataset.sample_ids:
        process_single_sample(
            dataset, sample_id, args.tile_size, args.save_dir, sdf_params=params,
        )


if __name__ == "__main__":
    main()
