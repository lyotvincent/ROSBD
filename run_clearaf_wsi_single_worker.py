import h5py
import tqdm
import argparse
import numpy as np
from pathlib import Path


from .dataset import OrionDataset, CyCIFDataset, BMCyCIFDataset, CometDataset, BaseDataset
from .baseline.clear_af import clear_af_2d
from .utils import read_multichannel_tiles, save_tile_outputs


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', choices=['single', 'cohort'], default='single')
    parser.add_argument('--dataset', choices=['orion', 'cycif', 'bmcycif', 'comet'], default='orion')
    parser.add_argument('--sample-id', nargs='+', default=[])
    parser.add_argument('--tile-size', type=int, default=None,
                        help='Tile size: 512 for comet, 256 for other datasets by default')
    parser.add_argument('--save-dir', required=True)
    return parser.parse_args()


def process_tile(tiles, af_ref):
    """
    Args:
        tiles: [N_c, H, W]
    Returns:
        result: [N_c, H, W]
    """
    raw = tiles.copy()
    result, bg = clear_af_2d(tiles, af_ref)
    return result, bg, raw


def process_wsi(slide_path_dict, af_path, coords, tile_size):
    for coord in tqdm.tqdm(coords, total=len(coords)):
        multichannel_tiles = read_multichannel_tiles(slide_path_dict, coord, tile_size)
        af_tile = read_multichannel_tiles({'AF1': af_path}, coord, tile_size)[0]
        try:
            result, bg, raw = process_tile(multichannel_tiles, af_tile)
        except Exception as error:
            raise RuntimeError(
                f"ClearAF failed at coord={tuple(map(int, coord))}, "
                f"channels={list(slide_path_dict)}, AF={af_path}"
            ) from error
        yield result, bg, raw


def process_single_sample(dataset: BaseDataset, sample_id, tile_size, save_dir):
    """Save result/bg/raw channel grids and result-left/raw-right PNGs.

    Files are saved to save_dir/sample_id/{result,bg,raw,result_raw_combinded}/
    x_y.png. Uses dataset.DEFAULT_CHANNELS and the configured AF reference.
    PNG previews clip to [0, 255] without per-image normalization.
    Returns the sample output directory.
    """
    if dataset.af is None:
        raise ValueError("ClearAF requires a configured AF reference; bmcycif and comet are not supported")
    slide_path_dict = dataset.get_single_multichannel(sample_id)
    chosen_slide_path_dict = {name: slide_path_dict[name] for name in dataset.DEFAULT_CHANNELS}
    with h5py.File(dataset.get_coord_path(sample_id, tile_size), "r") as file:
        coords = np.asarray(file["coords"])
    # if coords.ndim != 2 or coords.shape[1] != 2:
    #     raise ValueError("coords must have shape (N, 2)")
    # if not np.issubdtype(coords.dtype, np.integer):
    #     raise ValueError("coords must contain integer pixel coordinates")
    sample_dir = Path(save_dir) / sample_id
    af_path = slide_path_dict[dataset.af]
    outputs = process_wsi(chosen_slide_path_dict, af_path, coords, (tile_size, tile_size))
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
    if dataset.af is None:
        raise ValueError("ClearAF requires a configured AF reference; bmcycif and comet are not supported")
    if args.task == "single" and not args.sample_id:
        raise ValueError("--sample-id is required for --task single")
    for sample_id in args.sample_id or dataset.sample_ids:
        process_single_sample(
            dataset, sample_id, args.tile_size, args.save_dir,
        )


if __name__ == "__main__":
    main()
