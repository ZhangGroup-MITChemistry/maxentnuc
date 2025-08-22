from maxentnuc.simulation.model import PolymerModel, write_psf
import MDAnalysis as mda
import numpy as np
import pandas as pd
import tarfile
from tqdm import tqdm


def positions_to_distances(positions):
    # positions: (# conformations, [new dim 1], # loci, [new dim 2],  3)
    return np.linalg.norm(np.expand_dims(positions, -3) - np.expand_dims(positions, -2), axis=-1)


def positions_to_average_distance(positions, average=np.nanmedian):
    return average(positions_to_distances(positions), axis=0)


def positions_to_contacts(positions, thresh):
    distances = positions_to_distances(positions)
    num = (distances < thresh) & ~np.isnan(distances)
    den = ~np.isnan(distances)
    return np.sum(num, axis=0) / np.sum(den, axis=0)


class TrackData:
    def __init__(self, chrom, regions, positions, brightness=None):
        self.chrom = chrom
        self.regions = regions
        self.positions = positions
        self.brightness = brightness
        self.shifted = False

    def __copy__(self):
        return TrackData(self.chrom, self.regions.copy(), self.positions.copy(),
                         self.brightness.copy() if self.brightness is not None else None)

    @classmethod
    def from_trace_core(cls, trace_core_csv, validate=False):
        df = cls._read_trace_core_csv(trace_core_csv)
        df['Trace_ID'] = df['Trace_ID'].astype(int)
        df['Chrom_Start'] = df['Chrom_Start']
        df['Chrom_End'] = df['Chrom_End']
        df['X'] = df['X'].astype(float)
        df['Y'] = df['Y'].astype(float)
        df['Z'] = df['Z'].astype(float)

        chrom = df['Chrom'].iloc[0]
        assert all(df['Chrom'] == chrom)

        regions = sorted(set(map(tuple, df[['Chrom_Start', 'Chrom_End']].to_numpy().astype(int))))

        positions = np.zeros((max(df.Trace_ID)+1, len(regions), 3))
        positions[:] = np.nan
        for _, row in df.iterrows():
            region = (row.Chrom_Start, row.Chrom_End)
            if region not in regions:
                continue
            if validate:
                assert np.isnan(positions[row.Trace_ID, regions.index((row.Chrom_Start, row.Chrom_End))]).all(), \
                    f'No duplicate entries: {row.Trace_ID}: {regions.index((row.Chrom_Start, row.Chrom_End))}'
            positions[row.Trace_ID, regions.index((row.Chrom_Start, row.Chrom_End))] = row[['X', 'Y', 'Z']]

        return TrackData(chrom, regions, positions)

    @classmethod
    def _read_trace_core_csv(cls, trace_core_csv):
        with open(trace_core_csv) as f:
            for line in f:
                if 'columns=' in line:
                    columns = line.split('(')[1].split(')')[0].split(',')
                    columns = [x.strip() for x in columns]
                    break
            else:
                columns = None

        df = pd.read_csv(trace_core_csv, comment='#', names=columns, header=None, low_memory=False)
        df['Trace_ID'] = pd.factorize(df['Trace_ID'])[0]
        df = df.loc[df.Trace_ID != -1]
        df = df.loc[df.Chrom_End != 0]
        return df

    @classmethod
    def from_dipc(cls, path, chrom, start, end, test=False, scale=1000):
        tar = tarfile.open(path, "r:*")
        sub_paths = tar.getnames()
        if test:
            sub_paths = sub_paths[:10]

        dfs = []
        for sub_path in tqdm(sub_paths):
            df = pd.read_csv(
                tar.extractfile(sub_path),
                sep='\t',
                header=None,
                encoding='utf-8',
                comment='#',
                names=['Chromosome', 'Genomic_Index', 'x', 'y', 'z'],
                compression='gzip',
            )

            for a in ['a', 'b']:
                d = df.loc[(df.Chromosome == chrom + a) & (df.Genomic_Index >= start) & (df.Genomic_Index <= end)]
                d.loc[:, 'Chromosome'] = chrom
                d = d.set_index(['Chromosome', 'Genomic_Index'])
                d *= scale
                dfs += [d]

        assert all(df.index.equals(dfs[0].index) for df in dfs)
        regions = dfs[0].index.get_level_values('Genomic_Index')
        interval = regions[1] - regions[0]
        regions = [(s, s + interval) for s in regions]
        coords = np.array([df[['x', 'y', 'z']].to_numpy() for df in dfs])
        return TrackData(chrom, regions, coords)

    @classmethod
    def get_common_regions(cls, tracks):
        regions = sorted(set([region for track in tracks for region in track.regions]))
        return [region for region in regions if all(region in track.regions for track in tracks)]

    @classmethod
    def get_intersection(cls, tracks):
        start = max(track.regions[0][0] for track in tracks)
        end = min(track.regions[-1][1] for track in tracks)
        return start, end

    @classmethod
    def join(cls, tracks):
        chrom = tracks[0].chrom
        assert all(track.chrom == chrom for track in tracks)
        regions = tracks[0].regions
        assert all(track.regions == regions for track in tracks)
        positions = np.vstack([track.positions for track in tracks])
        if all(track.brightness is not None for track in tracks):
            brightness = np.vstack([track.brightness for track in tracks])
        else:
            brightness = None
        return TrackData(chrom, regions, positions, brightness)

    def shift_coordinates(self, shift):
        assert not self.shifted, 'Coordinates have already been shifted'
        self.shifted = True
        self.regions = [(x + shift, y + shift) for x, y in self.regions]

    def select_region(self, start, end):
        index = [i for i, (x, y) in enumerate(self.regions) if start <= x <= end and start <= y <= end]
        positions = self.positions[:, index]
        regions = [self.regions[i] for i in index]
        if self.brightness is not None:
            brightness = self.brightness[:, index]
        else:
            brightness = None
        return TrackData(self.chrom, regions, positions, brightness)

    def coarsen(self, regions):
        positions = np.zeros((self.positions.shape[0], len(regions), 3))
        for i, region in enumerate(regions):
            w = 0
            for j, (s, e) in enumerate(self.regions):
                assert (e - s) < (region[1] - region[0]), 'Attempting to coarsen but target is smaller than source'
                s_in_region = region[0] <= s <= region[1]
                e_in_region = region[0] <= e <= region[1]
                if s_in_region and e_in_region:
                    _w = 1
                elif s_in_region and not e_in_region:
                    _w = (e - region[0]) / (e - s)
                elif not s_in_region and e_in_region:
                    _w = (region[1] - s) / (e - s)
                else:
                    _w = 0
                w += _w
                positions[:, i] += _w * self.positions[:, j]
            positions[:, i] /= w
        return TrackData(self.chrom, regions, positions)

    def x(self):
        return np.array([(x + y) / 2 for x, y in self.regions])

    def is_compact(self):
        for i in range(len(self.regions) - 1):
            if self.regions[i][1] + 1 < self.regions[i + 1][0]:
                print(f'Non-compact region: {self.regions[i]}')
                return False
        return True

    def write_trajectory(self, name, nan_fraction=0.05):
        positions = self.positions.copy()
        positions = self.positions[np.mean(np.isnan(positions[:, :, 0]), axis=1) < nan_fraction]
        positions -= np.nanmean(positions, axis=1, keepdims=True)
        positions[np.isnan(positions)] = 0

        u = mda.Universe.empty(positions.shape[1], trajectory=True).load_new(positions)
        sele = u.select_atoms('all')
        with mda.Writer(f"{name}.dcd", n_atoms=positions.shape[1], multiframe=True) as writer:
            for ts in u.trajectory:
                writer.write(sele)

        model = PolymerModel(positions.shape[1])
        model.define_topology()
        write_psf(model.topology, f'{name}.psf')
