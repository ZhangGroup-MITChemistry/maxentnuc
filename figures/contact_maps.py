import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from scipy.ndimage import gaussian_filter1d

from neighbor_balance.neighbor import normalize_contact_map_average, normalize_contact_map_neighbor
from neighbor_balance.plotting import ContactMap, parse_region, format_ticks, get_epigenetics
from neighbor_balance.smoothing import coarsen_contact_map


class ChromHMM:
    states = [
        'Tss', 'TssFlnkU', 'TssFlnkD', 'TssFlnk', 
        'Enh1', 'Enh2', 'EnhG1', 'EnhG2',
        'Tx', 'TxWk', 'Biv',
        'ZNF/Rpts','ReprPC', 'Het', 'Quies'
    ]
    def __init__(self, bed_file, states=None):
        self.df = pd.read_csv(bed_file, sep='\t', header=None, names=['chrom', 'start', 'end', 'state', 'a', 'b', 'c', 'd', 'color'])
        if states is not None:
            self.states = states

        for state in self.states:
            if state not in self.df['state'].values:
                raise ValueError(f'State {state} not found in ChromHMM file.')
        for state in self.df['state'].unique():
            if state not in self.states:
                raise ValueError(f'State {state} found in ChromHMM file but not in provided states list.')

    def state_color(self, color):
        return tuple(map(lambda x: int(x) / 255, color.split(',')))

    def plot(self, chrom, start, end, ax=None, y=0, height=1):
        if ax is None:
            ax = plt.gca()
        region_df = self.df[(self.df['chrom'] == chrom) & (self.df['start'] < end) & (self.df['end'] > start)]
        for _, row in region_df.iterrows():
            color = self.state_color(row['color'])
            ax.fill_between([row['start'], row['end']], y - height / 2, y + height / 2, color=color, alpha=1)

    def legend(self):
        f, ax = plt.subplots(figsize=(2, len(self.states) * 0.2))
        for y, state in enumerate(self.states[::-1]):
            color = self.state_color(self.df[self.df['state'] == state].iloc[0]['color'])
            plt.plot([0, 1], [y, y], lw=6, solid_capstyle='butt', color=color)
            ax.text(1.1, y, state, va='center')
        ax.set_xlim(-0.5, 2)
        ax.axis('off')
        return f, ax


def load_region_contact_map(condition, name, coarsen=False):
    if condition == 'MESC':
        path = f'/home/joepaggi/maxentnuc-private/figures/region_contact_maps/WT_merged/{name}.npz'
    elif condition == 'G1E':
        print('WARNING: using MESC path for G1E')
        path = f'/home/joepaggi/maxentnuc-private/figures/region_contact_maps/WT_merged/{name}.npz'
    else:
        path = f'/orcd/data/binz/001/belong/for_revisions/{condition}/{condition}/{condition}_output_{name}.npz'
    if not os.path.exists(path):
        print('file does not exist:', path)
        return
    loaded = np.load(path, allow_pickle=True)
    contact_map = loaded['array']
    metadata = loaded['metadata'].item()
    chrom, start, end = parse_region(metadata['region'])

    def process(contact_map):
        if coarsen:
            if condition == 'HCT116' and name == '5':
                contact_map = coarsen_contact_map(contact_map, offset=6, base=4)
            else:
                contact_map = coarsen_contact_map(contact_map)
        return ContactMap(contact_map, chrom, start, end, 200)


    neighbor_contact_map = normalize_contact_map_neighbor(contact_map.copy())
    neighbor_contact_map = process(neighbor_contact_map)

    ice_contact_map = normalize_contact_map_average(contact_map.copy())
    ice_contact_map = process(ice_contact_map)
    return neighbor_contact_map, ice_contact_map


class Track:
    def __init__(self, path=None, chrom=None, x=None, values=None, bin=True, smoothing=200,
                 height=None, color='green', neg_color='red', name=None, ylims=None, style='fill',
                 ref_val=None):

        # Specify either path or (chrom, x, values)
        if path is None:
            assert chrom is not None
            assert x is not None and values is not None
        else:
            assert chrom is None
            assert x is None and values is None

        self.path = path
        self.chrom = chrom
        self.x = x
        self.values = values
        self.bin = bin
        self.smoothing = smoothing
        self.height = height
        self.color = color
        self.neg_color = neg_color
        self.name = name
        self.ylims = ylims
        self.style = style
        self.ref_val = ref_val

    def load_from_path(self, chrom, start, end):
        x, values = get_epigenetics(self.path, chrom, start, end,
                                    smoothing=self.smoothing, bin=self.bin)
        return x, values

    def plot(self, ax, chrom, start, end):
        if self.path is not None:
            x, values = self.load_from_path(chrom, start, end)
        else:
            assert self.chrom == chrom
            x, values = self.x, self.values

        if self.style == 'fill':
            ax.fill_between(x, 0, values, where=(values > 0), color=self.color, lw=0.5, alpha=1.0)
            ax.fill_between(x, 0, values, where=(values < 0), color=self.neg_color, lw=0.5, alpha=1.0)
            ax.axhline(0, c='black', lw=0.5)
        elif self.style == 'line':
            ax.plot(x, values, color=self.color)
        else:
            raise ValueError(f'Unknown style: {self.style}')

        if self.name is not None:
            ax.set_ylabel(self.name, rotation=0, ha='right', va='center')
        ax.yaxis.tick_right()
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_visible(False)

        if self.ref_val is not None:
            ax.axhline(self.ref_val, ls='--', c='grey', alpha=0.5)
            ax.set_yticks([self.ref_val])
    
        if self.ylims is not None:
            ax.set_ylim(*self.ylims)
            if self.ylims[0] == 0 and self.ref_val is None:
                ax.get_yticklabels()[0].set_visible(False)


def epigenetics_plot(self, tracks, depth=None, vmin=1e-4, smoothing=200, bin=True, show_map=True,
                     width=20, contact_height=0.75, track_height=0.5, mean_density=None):
    if depth is None:
        depth = (self.end - self.start) / 3

    height_ratios = []
    if show_map:
        contact_map_height = width * depth / (1.5 * (self.end - self.start))
        height_ratios += [contact_map_height]

    height_ratios += [contact_height]

    for track in tracks:
        height = track.height if track.height is not None else track_height
        height_ratios += [height]
        
    f, axs = plt.subplots(len(height_ratios), 1, figsize=(width, sum(height_ratios)), sharex=True,
                          gridspec_kw={'height_ratios': height_ratios})
    if show_map:
        self.plot_contact_map_horizontal(ax=axs[0], depth=depth, vmin=vmin, colorbar=False)
        contact_map_ax = axs[0]
        axs = axs[1:]
    else:
        contact_map_ax = None
        axs = axs

    marginal = self.get_marginal()
    marginal = gaussian_filter1d(marginal, sigma=5)
    if mean_density is None:
        mean_density = np.nanmean(marginal)
    marginal = Track(chrom=self.chrom, x=self.x(), values=marginal, color='black', style='line', name='Contact\ndensity', ylims=(0, 70), ref_val=mean_density)
    marginal.plot(ax=axs[0], chrom=self.chrom, start=self.start, end=self.end)

    for i, track in enumerate(tracks):
        ax = axs[i+1]
        track.plot(ax, chrom=self.chrom, start=self.start, end=self.end)

    axs[-1].set_xlim(self.start, self.end)
    format_ticks(axs[-1], y=False, rotate=False)

    return f, axs, contact_map_ax


def shrink_ax(ax, scale):
    pos = ax.get_position()
    ax.set_position([
        pos.x0,
        pos.y0 + pos.height * (1 - scale) / 2,
        pos.width,
        pos.height * scale
    ])


def plot_contact_map(cmap, zoom=None):
    f, axs = plt.subplots(2, 2, figsize=(11, 12), sharex='col',
                            gridspec_kw={'width_ratios': [40, 1], 'height_ratios': [10, 1],
                                        'wspace': 0.02, 'hspace': 0.015})

    im = cmap.plot_contact_map(ax=axs[0, 0], colorbar=False)
    plt.colorbar(im, cax=axs[0, 1], extend='min')
    shrink_ax(axs[0, 1], 0.66)
    
    #axs[1, 0].plot(ice_contact_map.x(), ice_contact_map.get_marginal(), c='gray')
    axs[1, 0].plot(cmap.x(), cmap.get_marginal(), c='black')
    axs[1, 0].axhline(40, ls='--', c='gray', lw=1)
    axs[1, 0].set_ylim(0, 70)

    axs[1, 1].axis('off')
    format_ticks(axs[1, 0], y=False, rotate=False)

    if zoom is not None:
        axs[0, 0].set_xlim(*zoom)
        axs[0, 0].set_ylim(*zoom[::-1])
        axs[1, 0].set_xlim(*zoom)
    return f, axs


def analyze_region(neighbor_contact_map, ice_contact_map, tracks, chromhmm_bed_file, condition, name, track_ylims, zoom=None):
    # f, axs = plot_contact_map(neighbor_contact_map, zoom=zoom)
    # axs[0, 0].set_title(f'Neighbor balanced contact map: {condition} region {name}')
    # plt.savefig(f'img/contact_map_neighbor_{condition}_{name}.png')
    # plt.close()

    # f, axs = plot_contact_map(ice_contact_map, zoom=zoom)
    # axs[0, 0].set_title(f'ICE balanced contact map: {condition} region {name}')
    # plt.savefig(f'img/contact_map_ice_{condition}_{name}.png')
    # plt.close()

    f, ax, _ = epigenetics_plot(neighbor_contact_map, _tracks,
                                show_map=False, width=10, track_height=0.20, contact_height=.8, mean_density=40)
    #ax[0].set_title(f'{condition} region {name}')
    for axis in ax:
        axis.spines['bottom'].set_linewidth(0.5)
        axis.xaxis.set_tick_params(width=0.5)
    
    if zoom is not None:
        ax[0].set_xlim(*zoom)

    plt.savefig(f'img/epigenetics_{condition}_{name}.png')
    plt.close()

if __name__ == '__main__':
    # chromhmm = {
    #     'GM12878': 'ENCFF140VIG',
    #     'H1': 'ENCFF008ABY',
    #     'K562': 'ENCFF106BGJ',
    #     'HCT116': 'ENCFF283KBS'}

    # for name, file in chromhmm.items():
    #     print(f'wget https://www.encodeproject.org/files/{file}/@@download/{file}.bed.gz')

    # chromhmm = {name: f'../../{file}.bed.gz' for name, file in chromhmm.items()}
    # chromhmm_plotter = ChromHMM(chromhmm['GM12878'])
    # f, ax = chromhmm_plotter.legend()
    # plt.savefig('img/chromhmm_legend.png')
    # plt.close()

    root = '/orcd/data/binz/001/belong/for_revisions/genomics_data'
    tracks = {
        'HCT116': {
            'A/B': '4DNFIUDME6HK.bw',
            'H3K4me3': 'ENCFF983JLZ.bigWig',
            'H3K27ac': 'ENCFF758DHJ.bigWig',
            'ATAC': 'ENCFF624HRW.bigWig',
            'RNA': 'ENCFF048EJE.bigWig',
            'H3K27me3': 'ENCFF232QSG.bigWig',
            'H3K9me3': 'ENCFF020CHJ.bigWig',
        },
        'H1': {
            'A/B': '4DNFIAO55M9G.bw',
            'H3K4me3': 'ENCFF180WKT.bigWig',
            'H3K27ac': 'ENCFF103PND.bigWig',
            'ATAC': '4DNFICPNO4M5.bw',
            #'RNA': 'ENCFF048EJE.bigWig',
            'H3K27me3': 'ENCFF927FVH.bigWig',
            'H3K9me3': 'ENCFF385ZBQ.bigWig',
            'LaminB1 DamID': '4DNFIXNBG8L1.bw',
            'SON TSA-seq': '4DNFI625PP2A.bw',
            'Condensability': 'condensibility_H1.bw',
        },
        'GM12878': {
            'A/B': '4DNFI3VXZ48N.bw',
            'H3K4me3': 'ENCFF975ARJ.bigWig',
            'H3K27ac': 'ENCFF798KYP.bigWig',
            'ATAC': 'ENCFF180ZAY.bigWig',
            #'RNA': 'ENCFF048EJE.bigWig',
            'H3K27me3': 'ENCFF677PYB.bigWig',
            'H3K9me3': 'ENCFF701GHA.bigWig',
            'Condensability': 'condensibility_GM12878.bw',
        },
        'K562': {
            'A/B': '4DNFIWUAO2QI.bw',
            'H3K4me3': 'ENCFF767UON.bigWig',
            'H3K27ac': 'ENCFF469JMR.bigWig',
            'ATAC': 'ENCFF357GNC.bigWig',
            #'RNA': 'ENCFF048EJE.bigWig',
            'H3K27me3': 'ENCFF847BFA.bigWig',
            'H3K9me3': 'ENCFF330EOT.bigWig',
            #'H3K4me1': 'ENCFF287LBI.bigWig',
        },
    }

    _tracks = {}
    for condition in tracks:
        _tracks[condition] = {}
        for name, path in tracks[condition].items():
            if name == 'Condensability':
                _tracks[condition][name] = path
            else:
                _tracks[condition][name] = f'{root}/{condition}/{path}'
    tracks = _tracks

    supercloud = '/home/joepaggi/orcd/pool'
    tracks['MESC'] = {
            'A/B': f'{supercloud}/omics/genomics_data/4DNFI14DZTHU.bw',
            'SON': f'{supercloud}/omics/genomics_data/4DNFI14DZTHU.bw',
            'LaminB1': f'{supercloud}/omics/genomics_data/4DNFIGYZ56AD.bw',
            'H1': f'{supercloud}/omics/genomics_data/GSM1124783_H1d-1_IP-IN.mm39.bw',
            'H3K27ac': f'{supercloud}/omics/new_genomics_data/ENCFF230RNU.mm39.bw',
            'H3K4me3': f'{supercloud}/omics/new_genomics_data/ENCFF523UIR.mm39.bw',
            'H3K27me3': f'{supercloud}/omics/new_genomics_data/ENCFF160FEV.mm39.bw',
            'H3K9me3': f'{supercloud}/omics/new_genomics_data/ENCFF293DGT.mm39.bw',

            #'H3K4me1': f'{supercloud}/omics/new_genomics_data/ENCFF410CGG.mm39.bw',
            #'ATAC': f'{supercloud}/omics/genomics_data/GSE98390_E14_ATAC_MERGED.DANPOS.mm39.bw',
            # 'CTCF': f'{supercloud}/genomics_data/GSM2418860_WT_CTCF.mm39.bw',
            # 'SMC1A': f'{supercloud}/genomics_data/GSM3508477_C59_Smc1a_SpikeInNormalized.mm39.bw',
            # 'MED1': f'{supercloud}/genomics_data/GSM560347_10022009_42TM0AAXX_B6.mm39.bw',
            # 'RING1B': f'{supercloud}/genomics_data/GSE96107_ES_Ring1B.mm39.bw',
            #'POLII': f'{supercloud}/omics/genomics_data/GSM6809981_WT_PolII_xChIP_r2_mm39_MERGED_rmdup_downsampled.bw',
            #'RNA-seq': f'{supercloud}/omics/genomics_data/GSE123636_C59_1_2_RNAseq_coverage.mm39.bw',
        }



    track_ylims = {
        'Contact\ndensity': (0, 70),
        'A/B': (-1, 1),
        'H3K4me3': (0, 5),
        'H3K27ac': (0, 20),
        'ATAC': (0, 10),
        'RNA': (0, 5),
        'H3K27me3': (0, 10),
        'H3K9me3': (0, 5),
        'LaminB1 DamID': (-2, 2),
        'SON TSA-seq': (-2, 2),
        'Condensability': (-2, 4),
    }

    """
    1 chr6:25,112,000-28,621,800
    3 chr5:157,036,200-160,147,400
    4 chr1:207,626,400-210,338,200
    5 chr4:61,369,200-64,434,200
    """

    conditions = ['MESC', 'GM12878', 'H1', 'HCT116', 'K562']
    regions = ['1', '3', '4', '5']#, '6', '7', '8', '9', '10', '11', '12', '13', '14']

    zooms = {
        '1': {
            'zoom': (26_000_000, 28_000_000),
            'zoomup': (26_000_000, 26_250_000),
            'zoomupup': (26_000_000, 26_600_000),
            'zoomdown': (27_100_000, 27_950_000),
        },
        '3': {
            'zoom': (157_100_000, 159_150_000),
        },
        '5': {
            'zoom': (62_000_000, 64_000_000)
        }
    }

    from scipy.stats import gaussian_kde
    def myhist(data, bins, ax=None, density=True, bw = 1.0, alpha=0.75, lw=3, weights=None, **kwargs):
        p = plt if ax is None else ax
        if data.min() < bins[0] or data.max() > bins[-1]:
            print(f'Warning: data outside range ({data.min()} to {data.max()} vs {bins[0]} to {bins[-1]})')
        
        cov = np.cov(data, aweights=weights if weights is not None else None)
        kde = gaussian_kde(data, weights=weights, bw_method=bw / cov**0.5)

        y = kde(bins)
        p.plot(bins, y, lw=lw, alpha=alpha, **kwargs)


    for condition in conditions:
        buffer = 100_000 // 200
        
        all_marginal = []
        all_acetyl = []
        all_methyl = []
        all_k9methyl = []
        all_ab = []
        all_condensability = []
        if condition == 'MESC':
            _regions = ['ppm1g', 'klf1', 'sox2', 'fbn2', 'nanog']
        else:
            _regions = regions
        for name in _regions:
            neighbor_contact_map, ice_contact_map = load_region_contact_map(condition, name, coarsen=False)
            marginal = neighbor_contact_map.get_marginal()
            x = neighbor_contact_map.x()-100

            _x, acetyl = get_epigenetics(tracks[condition]['H3K27ac'], neighbor_contact_map.chrom,
                                        neighbor_contact_map.start, neighbor_contact_map.end, bin=True)
            assert np.all(x == _x), 'x coordinates do not match'

            _x, methyl = get_epigenetics(tracks[condition]['H3K27me3'], neighbor_contact_map.chrom,
                                        neighbor_contact_map.start, neighbor_contact_map.end, bin=True)
            assert np.all(x == _x), 'x coordinates do not match'

            _x, k9methyl = get_epigenetics(tracks[condition]['H3K9me3'], neighbor_contact_map.chrom,
                                        neighbor_contact_map.start, neighbor_contact_map.end, bin=True)
            assert np.all(x == _x), 'x coordinates do not match'

            _x, ab = get_epigenetics(tracks[condition]['A/B'], neighbor_contact_map.chrom,
                                        neighbor_contact_map.start, neighbor_contact_map.end, bin=True)
            assert np.all(x == _x), 'x coordinates do not match'

            if condition in ['H1', 'GM12878']:
                _x, condensability = get_epigenetics(tracks[condition]['Condensability'], neighbor_contact_map.chrom,
                                            neighbor_contact_map.start, neighbor_contact_map.end, bin=True)
                assert np.all(x == _x), 'x coordinates do not match'
                all_condensability.append(condensability[buffer:-buffer])

            all_marginal.append(marginal[buffer:-buffer])
            all_acetyl.append(acetyl[buffer:-buffer])
            all_methyl.append(methyl[buffer:-buffer])
            all_k9methyl.append(k9methyl[buffer:-buffer])
            all_ab.append(ab[buffer:-buffer])
            
        all_marginal = np.concatenate(all_marginal)
        all_acetyl = np.concatenate(all_acetyl)
        all_methyl = np.concatenate(all_methyl)
        all_k9methyl = np.concatenate(all_k9methyl)
        all_ab = np.concatenate(all_ab)
        

        acetyl_cut = np.percentile(all_acetyl, 99)
        methyl_cut = np.percentile(all_methyl, 99)
        k9methyl_cut = np.percentile(all_k9methyl, 99)

        acetyl_cut_low = np.percentile(all_acetyl, 80)
        methyl_cut_low = np.percentile(all_methyl, 80)
        k9methyl_cut_low = np.percentile(all_k9methyl, 80)

        if condition in ['H1', 'GM12878']:
            all_condensability = np.concatenate(all_condensability)
            f, ax = plt.subplots(figsize=(4, 2))
            bw = 0.2
            bins = np.linspace(-3, 5, 301)
            myhist(all_condensability[all_acetyl > acetyl_cut], bins=bins, bw=bw, label='H3K27ac high')
            myhist(all_condensability[all_methyl > methyl_cut], bins=bins, bw=bw, label='H3K27me3 high')
            myhist(all_condensability[all_k9methyl > k9methyl_cut], bins=bins, bw=bw, label='H3K9me3 high')
            myhist(all_condensability[(all_acetyl <= acetyl_cut_low) & (all_methyl <= methyl_cut_low) & (all_k9methyl <= k9methyl_cut_low)], bins=bins, bw=bw, label='All low')
            ax.legend()
            ax.set_ylim(0)
            ax.set_xlabel('Condensability')
            plt.savefig(f'img/condensability_histograms_{condition}.png')
            plt.close()

            f, ax = plt.subplots(figsize=(4,2))
            myhist(all_marginal[(all_ab < 0) & (all_acetyl > acetyl_cut)], bins=bins, bw=bw, label='B, H3K27ac high', density=True, color='purple')
            myhist(all_condensability[(all_ab < 0) & (all_acetyl < acetyl_cut_low)], bins=bins, bw=bw, label='B, H3K27ac low', density=True, color='red')
            myhist(all_condensability[(all_ab >= 0) & (all_acetyl > acetyl_cut)], bins=bins, bw=bw, label='A, H3K27ac high', density=True, color='green')
            myhist(all_condensability[(all_ab >= 0) & (all_acetyl < acetyl_cut_low)], bins=bins, bw=bw, label='A, H3K27ac low', density=True, color='blue')
            ax.legend()
            ax.set_ylim(0)
            ax.set_xlabel('Condensability')
            plt.savefig(f'img/condensability_histograms_ab_{condition}.png')
            plt.close()

        f, ax = plt.subplots(figsize=(4, 2))
        bw = 1.0
        bins = np.linspace(0, 60, 301)
        if condition == 'K562':
            bins = np.linspace(0, 80, 401)
        myhist(all_marginal[all_acetyl > acetyl_cut], bins=bins, label='H3K27ac high')
        myhist(all_marginal[all_methyl > methyl_cut], bins=bins, label='H3K27me3 high')
        myhist(all_marginal[all_k9methyl > k9methyl_cut], bins=bins, label='H3K9me3 high')
        myhist(all_marginal[(all_acetyl < acetyl_cut_low) & (all_methyl < methyl_cut_low) & (all_k9methyl < k9methyl_cut_low)], bins=bins, label='All low')
        ax.legend()
        ax.set_ylim(0)
        ax.set_xlabel('Contact density')
        plt.savefig(f'img/marginal_histograms_{condition}.png')
        plt.close()

        f, ax = plt.subplots(figsize=(4,2))
        myhist(all_marginal[(all_ab < 0) & (all_acetyl > acetyl_cut)], bins=bins, label='B, H3K27ac high', density=True, color='purple')
        myhist(all_marginal[(all_ab < 0) & (all_acetyl < acetyl_cut_low)], bins=bins, label='B, H3K27ac low', density=True, color='red')
        myhist(all_marginal[(all_ab >= 0) & (all_acetyl > acetyl_cut)], bins=bins, label='A, H3K27ac high', density=True, color='green')
        myhist(all_marginal[(all_ab >= 0) & (all_acetyl < acetyl_cut_low)], bins=bins, label='A, H3K27ac low', density=True, color='blue')
        ax.legend()
        ax.set_ylim(0)
        ax.set_xlabel('Contact density')
        plt.savefig(f'img/marginal_histograms_ab_{condition}.png')
        plt.close()


    f, ax = plt.subplots()
    for condition in ['H1', 'GM12878']:
        for name in regions:
            neighbor_contact_map, ice_contact_map = load_region_contact_map(condition, name, coarsen=False)
            marginal = neighbor_contact_map.get_marginal()
            x = neighbor_contact_map.x()-100

            track_name = 'Condensability'
            _x, values = get_epigenetics(tracks[condition][track_name], neighbor_contact_map.chrom,
                                        neighbor_contact_map.start, neighbor_contact_map.end, bin=True)
            print(x[:4], _x[:4])
            assert np.all(x == _x), 'x coordinates do not match'

            ax.scatter(values, marginal, label=condition, alpha=0.1, s=1)
    ax.legend()
    ax.set_xlabel('Condensability score')
    ax.set_ylabel('Contact density')
    plt.savefig('img/condensability_vs_contact_density.png')
    plt.close()

    f, ax = plt.subplots(8, figsize=(10, 3), sharex=True, gridspec_kw={'height_ratios': [2, 1]*4})
    name = '1'
    zoom_start, zoom_end = 26_000_000, 26_300_000
    for i, condition in enumerate(conditions):
        neighbor_contact_map, ice_contact_map = load_region_contact_map(condition, name, coarsen=False)
        marginal = neighbor_contact_map.get_marginal()
        ax[2*i].plot(neighbor_contact_map.x(), marginal, label=condition, color='black')
        ax[2*i].axhline(40, ls='--', c='gray', lw=1)
        ax[2*i].set_yticks([40])
        ax[2*i].set_ylim(0, 70)
        ax[2*i].set_ylabel(condition, rotation=0, ha='right', va='center')

        track_name = 'H3K27ac'
        x, values = get_epigenetics(tracks[condition][track_name], neighbor_contact_map.chrom,
                                    zoom_start, zoom_end,
                                    smoothing=200, bin=True)
        values /= np.percentile(values, 95)
        ax[2*i+1].fill_between(x, 0, values, label=condition, alpha=0.5, color='green')
        ax[2*i+1].set_yticks([1])
        
    for a in ax:
        a.yaxis.tick_right()
        a.spines['top'].set_visible(False)
        a.spines['right'].set_visible(False)
        a.spines['left'].set_visible(False)
    ax[-1].set_xlim(zoom_start, zoom_end)
    format_ticks(ax[-1], y=False, rotate=False)
    # ax[0].legend()
    # ax[1].set_ylim(0)
    # ax[0].yaxis.tick_right()
    # ax[1].yaxis.tick_right()
    # ax[1].set_ylabel('H3K27ac\n(normalized)', rotation=0, ha='right', va='center')
    plt.savefig('img/compare_region1_zoomup.png')
    plt.close()

    for condition in conditions:
        for name in regions:
            print(f'Processing condition {condition}, region {name}...')
            neighbor_contact_map, ice_contact_map = load_region_contact_map(condition, name, coarsen=False)
        
            _tracks = []
            if 'Condensability' in tracks[condition]:
                _tracks += [Track(path=tracks[condition]['Condensability'], style='line', name='Condensability', ylims=track_ylims['Condensability'], height=0.4, color='orange', ref_val=3.0)]

            for track_name in ['H3K27ac', 'H3K27me3', 'H3K9me3', 'A/B']:
                if track_name == 'H3K27ac':
                    color = 'green'
                elif track_name == 'H3K27me3':
                    color = 'purple'
                elif track_name == 'H3K9me3':
                    color = 'blue'
                elif track_name == 'A/B':
                    color = 'green'
                _tracks += [Track(path=tracks[condition][track_name], style='fill', name=track_name, ylims=track_ylims[track_name], color=color,
                                ref_val=0.0 if track_name == 'A/B' else None)]
        
            analyze_region(neighbor_contact_map, ice_contact_map, _tracks, None, condition, name, track_ylims)

    # for condition in conditions:
    #     for name, _zooms in zooms.items():
    #         neighbor_contact_map, ice_contact_map = load_region_contact_map(condition, name, coarsen=False)
    #         for zoom_name, zoom in _zooms.items():
    #             _name = f'{name}_{zoom_name}'
    #             print('Processing zoomed region:', condition, _name)
    #             analyze_region(neighbor_contact_map, ice_contact_map, _tracks, chromhmm[condition], condition, _name, track_ylims, zoom=zoom)
