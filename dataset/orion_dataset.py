import csv
from pathlib import Path
from .base_dataset import BaseDataset


class OrionDataset(BaseDataset):
    """Index Orion channel slides and patch-coordinate files.

    Attributes:
        slide_paths_dict: {sample_id: {channel_name: Path}}. Channels follow
            channels.tsv order, or filename order when no manifest exists.
        coord_path_dict: {tile_size: {sample_id: Path}}, with string size keys.
        DEFAULT_CHANNELS: Seven shared channels; CD11c is excluded because
            it is absent in eight samples. Equivalent early-sample names for
            selected channels are canonicalized without changing file paths.
    """

    def __init__(self, slide_root=None, coord_root=None):
        self.DEFAULT_CHANNELS = [
            "01_Nucleus_Hoechst", "05_CD68_ArgoFluor535", "06_CD163_ArgoFluor580L",
            "07_CD20_ArgoFluor660L", "08_CD4_ArgoFluor572",
            "09_CD8a_ArgoFluor602", "11_PD-L1_ArgoFluor662",
        ]
        self.af = '02_AF1'
        data_root = Path("/data/wzx/wala-christensen-2026-pik3ca-crc")
        self.slide_root = Path(slide_root) if slide_root is not None else data_root / "orion_sp"
        if coord_root is None:
            coord_root = {
                str(size): data_root / "orion_patch_info" / f"{size}_{size}"
                for size in (256, 512, 1024)
            }
        self.coord_root = {str(size): Path(root) for size, root in coord_root.items()}

        for root in (self.slide_root, *self.coord_root.values()):
            if not root.is_dir():
                raise FileNotFoundError(f"Dataset directory does not exist: {root}")

        channel_aliases = {
            "01_Hoechst": "01_Nucleus_Hoechst",
            "05_CD68_Argo535": "05_CD68_ArgoFluor535",
            "06_CD163_Argo580L": "06_CD163_ArgoFluor580L",
            "07_CD20_Argo660L": "07_CD20_ArgoFluor660L",
            "08_CD4_Argo572": "08_CD4_ArgoFluor572",
            "09_CD8a_Argo602": "09_CD8a_ArgoFluor602",
            "11_PD-L1_Argo662": "11_PD-L1_ArgoFluor662",
        }
        self.slide_paths_dict = {}
        for slide_dir in sorted(self.slide_root.iterdir()):
            if not slide_dir.is_dir():
                continue
            channel_paths = sorted(slide_dir.glob("*.ome.tif"))
            if not channel_paths:
                continue

            manifest = slide_dir / "channels.tsv"
            channels = {}
            if manifest.is_file():
                with manifest.open(newline="") as file:
                    for row in csv.DictReader(file, delimiter="\t"):
                        channel_name = channel_aliases.get(row["channel_name"], row["channel_name"])
                        channel_path = slide_dir / row["file"]
                        if not channel_path.is_file():
                            raise FileNotFoundError(f"Channel slide does not exist: {channel_path}")
                        if channel_name in channels:
                            raise ValueError(f"Duplicate channel {channel_name!r} in {manifest}")
                        channels[channel_name] = channel_path
            else:
                channels = {
                    path.name.removesuffix(".ome.tif"): path
                    for path in channel_paths
                }
            if not channels:
                raise ValueError(f"No channels listed in {manifest}")
            self.slide_paths_dict[slide_dir.name] = channels

        self.sample_ids = list(self.slide_paths_dict)
        self.coord_path_dict = {
            size: {
                path.stem: path
                for path in sorted(root.glob("*.h5"))
                if path.is_file()
            }
            for size, root in self.coord_root.items()
        }

    def get_single_multichannel(self, sample_id):
        """Return {channel_name: Path} for one sample, without opening slides."""
        return self.slide_paths_dict[sample_id].copy()

    def get_coord_path(self, sample_id, tile_size=256):
        """Return a sample's HDF5 coordinate path for an integer/string size."""
        return self.coord_path_dict[str(tile_size)][sample_id]
