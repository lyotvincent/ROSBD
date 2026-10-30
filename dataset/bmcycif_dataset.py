"""Index the flat, per-round/per-channel BM-Her2N75-15 TIFF collection."""

import re
from pathlib import Path

from .base_dataset import BaseDataset


class BMCyCIFDataset(BaseDataset):
    """Group registered TIFF files by scene, using physical channel IDs.

    sample_ids: Full IDs such as BM-Her2N75-15-Scene-017.
    slide_paths_dict: Physical IDs ("R1_c1", ..., "R5_c5").
    channel_info_dict: Round, channel number and panel text from filenames.
        Panel marker order is recorded without assuming a c1-c5 mapping.
    DEFAULT_CHANNELS: R1_c1-R1_c4, referring to round 1 channels 1-4.
        No AF channel is inferred; set af explicitly once identified.
    coord_path_dict: {size: {sample_id: HDF5 Path}}, using string size keys.
        Default folders are slide_root/patch_info/{size}_{size}; missing
        default folders yield empty indexes. Additional size_size folders are
        discovered automatically. Coordinate files are not created.
    available_tile_sizes: Integer sizes with at least one indexed HDF5 file.
    get_coords: Load level-0 (x,y) coordinates without dtype conversion and
        validate shape, integer/non-negative values and optional HDF5 metadata.
    """

    FILE_PATTERN = re.compile(
        r"Registered-R(?P<round>\d+)_(?P<panel>[^_]+)_"
        r"(?P<sample>.+-Scene-\d+)_c(?P<channel>\d+)_ORG\.tiff?",
        re.IGNORECASE,
    )

    def __init__(self, slide_root=None, coord_root=None, exclude_samples=None):
        self.DEFAULT_CHANNELS = [
            "R1_c1", "R1_c2", "R1_c3", "R1_c4", "R1_c5"
        ]
        self.slide_root = Path(slide_root) if slide_root is not None else Path(
            "/data/wzx/BM-Her2N75-15"
        )
        self.exclude_samples = set(exclude_samples or ())
        self.af = None
        default_coord_root = coord_root is None
        if coord_root is None:
            coord_root = {
                str(size): self.slide_root / "patch_info" / f"{size}_{size}"
                for size in (256, 512, 1024)
            }
        if default_coord_root:
            patch_info = self.slide_root / "patch_info"
            if patch_info.is_dir():
                for directory in sorted(patch_info.iterdir()):
                    match = re.fullmatch(r"([1-9]\d*)_([1-9]\d*)", directory.name)
                    if directory.is_dir() and match and match.group(1) == match.group(2):
                        coord_root[match.group(1)] = directory
        self.coord_root = {
            str(size): Path(root) for size, root in sorted(
                coord_root.items(), key=lambda item: int(item[0])
            )
        }
        records = []
        for path in self.slide_root.iterdir():
            if not path.is_file() or path.suffix.lower() not in (".tif", ".tiff"):
                continue
            match = self.FILE_PATTERN.fullmatch(path.name)
            if match is None:
                if path.name.startswith("Registered-R"):
                    raise ValueError(f"Unrecognized BM-CyCIF filename: {path.name}")
                continue
            fields = match.groupdict()
            sample_id = fields["sample"]
            if sample_id in self.exclude_samples:
                continue
            records.append((
                sample_id, int(fields["round"]), int(fields["channel"]),
                fields["panel"], path,
            ))

        self.slide_paths_dict = {}
        self.channel_info_dict = {}
        for sample_id, round_number, channel_number, panel, path in sorted(records):
            channel_name = f"R{round_number}_c{channel_number}"
            channels = self.slide_paths_dict.setdefault(sample_id, {})
            info = self.channel_info_dict.setdefault(sample_id, {})
            if channel_name in channels:
                raise ValueError(f"Duplicate channel {channel_name} for {sample_id}")
            channels[channel_name] = path
            info[channel_name] = {
                "round": round_number, "channel": channel_number,
                "panel_markers": panel.split("."),
            }

        self.sample_ids = list(self.slide_paths_dict)
        self.coord_path_dict = {
            size: {
                path.stem: path
                for path in sorted(root.glob("*.h5"))
                if path.is_file() and path.stem in self.slide_paths_dict
            }
            for size, root in self.coord_root.items()
        }

        self.available_tile_sizes = [
            int(size) for size, paths in self.coord_path_dict.items() if paths
        ]

    def get_single_multichannel(self, sample_id):
        """Return an independent mapping from physical channel IDs to TIFF paths."""
        return self.slide_paths_dict[sample_id].copy()

    def get_coord_path(self, sample_id, tile_size=256):
        """Return the indexed HDF5 coordinate path."""
        return self.coord_path_dict[str(tile_size)][sample_id]

    def get_coords(self, sample_id, tile_size=256):
        """Load an integer [N,2] array of level-0 (x,y) coordinates."""
        import h5py
        import numpy as np

        path = self.get_coord_path(sample_id, tile_size)
        with h5py.File(path, "r") as file:
            coords = np.asarray(file["coords"])
            stored_sample = file.attrs.get("sample_id")
            if isinstance(stored_sample, bytes):
                stored_sample = stored_sample.decode("utf-8")
            if stored_sample is not None and stored_sample != sample_id:
                raise ValueError(f"Sample ID mismatch in {path}: {stored_sample}")
            stored_size = file.attrs.get("patch_size")
            if stored_size is not None and int(stored_size) != int(tile_size):
                raise ValueError(f"Patch size mismatch in {path}: {stored_size}")
        if coords.ndim != 2 or coords.shape[1] != 2:
            raise ValueError(f"Expected coords [N,2] in {path}, got {coords.shape}")
        if not np.issubdtype(coords.dtype, np.integer) or np.any(coords < 0):
            raise ValueError(f"Coordinates must be non-negative integers in {path}")
        return coords
