from pathlib import Path
import h5py
import numpy as np
import pandas as pd

from .base_dataset import BaseDataset


class CometDataset(BaseDataset):

    CHANNELS = ['DAPI', 'CK8', 'VEGF', 'CD45RA', 'Vimentin', 'E-cadherin',
                'CD1c', 'IL-10', 'Podoplanin', 'FXIIIa', 'CD66', 'CD20',
                'αSMA', 'CD45', 'HepPar1', 'CD206']

    def __init__(self, slide_root=None, coord_root=None, annotation_root=None):
        root = Path(__file__).resolve().parents[1] / "datasets" / "COMET"
        self.RNA_MARKERS = {'CD20': 'MS4A1', 'CD1c': 'CD1C', 'CD206': 'MRC1',
                            'E-cadherin': 'CDH1', 'FXIIIa': 'F13A1', 'IL-10': 'IL10',
                            'Podoplanin': 'PDPN', 'HepPar1': 'CPS1'}
        self.DEFAULT_CHANNELS = [*self.RNA_MARKERS, 'DAPI']
        self.SCORE_CHANNELS = list(self.RNA_MARKERS)
        self.af = None
        self.slide_root = Path(slide_root) if slide_root else root/'COMET_ome_tiff'
        self.annotation_root = Path(annotation_root) if annotation_root else root/'marker_annotations/rna_seed_v1'
        if coord_root is None:
            coord_root = {p.name.split('_')[0]: p for p in sorted((root/'patch_info').glob('*_*')) if p.is_dir()}
        self.coord_root = {str(size): Path(path) for size, path in coord_root.items()}
        self.slide_paths_dict = {
            path.name.removesuffix('_comet.ome.tif'): {
                name: (path, index) for index, name in enumerate(self.CHANNELS)
            } for path in sorted(self.slide_root.glob('core*_comet.ome.tif'))
        }
        self.sample_ids = list(self.slide_paths_dict)
        self.coord_path_dict = {
            size: {p.stem: p for p in sorted(directory.glob('*.h5')) if p.stem in self.sample_ids}
            for size, directory in self.coord_root.items()
        }
        self.available_tile_sizes = [int(size) for size, paths in self.coord_path_dict.items() if paths]

    def get_single_multichannel(self, sample_id):
        return self.slide_paths_dict[sample_id].copy()

    def get_coord_path(self, sample_id, tile_size=512):
        return self.coord_path_dict[str(tile_size)][sample_id]

    def get_coords(self, sample_id, tile_size=512):
        with h5py.File(self.get_coord_path(sample_id, tile_size)) as handle:
            return np.asarray(handle['coords'])

    def get_annotation_tiles(self, sample_id=None):
        tiles = pd.read_csv(self.annotation_root/'tiles.csv')
        tiles['sample_id'] = tiles['core'].str.split('_').str[0]
        if sample_id is not None:
            tiles = tiles.loc[tiles.sample_id.eq(sample_id)]
        return tiles.reset_index(drop=True)

    def get_rna_paths(self, sample_id):
        root = self.slide_root.parent
        core = next((root/'UnumLocalia').glob(f'{sample_id}_*'))
        return {'core': core.name,
                'quantification': root/'Cell_quantifications'/f'{core.name}_quantification_cells.csv',
                'cell_boundaries': core/'xenium/cell_boundaries_comet_space.geojson'}
