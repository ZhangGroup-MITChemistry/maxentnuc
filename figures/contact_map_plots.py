from neighbor_balance.plotting import format_ticks, parse_region, apply_matplotlib_style, ContactMap, get_epigenetics, get_epigenetics_ylims
from neighbor_balance.smoothing import coarsen_contact_map
from neighbor_balance.neighbor import normalize_contact_map_neighbor, normalize_contact_map_average
import matplotlib.pyplot as plt
import numpy as np
import click


supercloud = '/home/joepaggi/orcd/pool/omics'
tracks = {
    'ATAC': f'{supercloud}/genomics_data/GSE98390_E14_ATAC_MERGED.DANPOS.mm39.bw',

    'H3K27ac': f'{supercloud}/new_genomics_data/ENCFF230RNU.mm39.bw',
    'H3K4me1': f'{supercloud}/new_genomics_data/ENCFF410CGG.mm39.bw',
    'H3K4me3': f'{supercloud}/new_genomics_data/ENCFF523UIR.mm39.bw',

    'H3K27me3': f'{supercloud}/new_genomics_data/ENCFF160FEV.mm39.bw',
    'H3K9me3': f'{supercloud}/new_genomics_data/ENCFF293DGT.mm39.bw',

    'H1': f'{supercloud}/genomics_data/GSM1124783_H1d-1_IP-IN.mm39.bw',
    'RCMC-pileup': f'{supercloud}/../rcmc/nucleosomes_pe/all.nodups.shift73.bw',

    'CTCF': f'{supercloud}/genomics_data/GSM2418860_WT_CTCF.mm39.bw',
    #'SMC1A': f'{supercloud}/genomics_data/GSM3508477_C59_Smc1a_SpikeInNormalized.mm39.bw',
    'MED1': f'{supercloud}/genomics_data/GSM560347_10022009_42TM0AAXX_B6.mm39.bw',
    #'RING1B': f'{supercloud}/genomics_data/GSE96107_ES_Ring1B.mm39.bw',
}


def load_maps(conditions, regions, root='region_contact_maps'):
    maps = {}
    for condition in conditions:
        for name in regions:
            print(condition, name)
            loaded = np.load(f'{root}/{condition}/{name}.npz', allow_pickle=True)
            contact_map = loaded['array']
            metadata = loaded['metadata'].item()
            chrom, start, end = parse_region(metadata['region'])
            if condition not in maps:
                maps[condition] = {}
            maps[condition][name] = {}

            maps[condition][name]['ice'] = ContactMap(normalize_contact_map_average(contact_map.copy()), chrom, start, end, 200)
            maps[condition][name]['neighbor'] = ContactMap(normalize_contact_map_neighbor(contact_map.copy()), chrom, start, end, 200)

            maps[condition][name]['ice'].contact_map = coarsen_contact_map(maps[condition][name]['ice'].contact_map)
            maps[condition][name]['neighbor'].contact_map = coarsen_contact_map(maps[condition][name]['neighbor'].contact_map)
    return maps


@click.command()
@click.argument('condition', default='WT_merged')
@click.option('--root', default='region_contact_maps', help='Root directory for contact maps.')
def main(condition, root):
    regions = ['ppm1g'] #['sox2', 'klf1', 'ppm1g', 'fbn2', 'nanog']

    maps = load_maps([condition], regions, root=root)

    # Plots for full regions.
    def plot_full(cmap1, cmap2):
        return cmap2.compare(cmap1, vmin=1e-4, vmax=1e0)

    for name in regions:
        f, ax = plot_full(maps[condition][name]['ice'], maps[condition][name]['neighbor'])
        ax[0, 0].set_title('ICE')
        ax[0, 3].set_title('Neighbor')
        ax[0, 6].set_title('Log2 Neighbor / ICE')
        plt.savefig(f'img/contact_map_{condition}_{name}.pdf')
        plt.close()

    # Plots for zoomed regions.
    zooms = {
        'ppm1g zoom': 'chr5:31,330,000-31,430,000',
        'sox2 zoom': 'chr6:34,650,000-34,850,000',
        'sox2 zoom 2': 'chr3:34,000,000-34,150,000',
        'klf1 zoom': 'chr8:85,500,000-85,700,000',
        'ppm1g zoom 2': 'chr5:31,930,000-32,030,000',
        'ppm1g zoom 3': 'chr5:31,730,000-31,830,000',
        'ppm1g zoom 4': 'chr5:31,700,000-32,100,000',
    }

    def plot_zoom(cmap1, cmap2, start, end):
        return cmap2.compare(cmap1, zoom_start=start, zoom_end=end, vmin=1e-3, bw=2)

    for name, region in zooms.items():
        chrom, start, end = parse_region(region)
        full_region_name = name.split(' ')[0]
        if full_region_name not in regions:
            continue

        f, ax = plot_zoom(maps[condition][full_region_name]['ice'], maps[condition][full_region_name]['neighbor'], start, end)
        ax[0, 0].set_title('ICE')
        ax[0, 3].set_title('Neighbor')
        ax[0, 6].set_title('Log2 Neighbor / ICE')
        plt.savefig(f'img/contact_map_{condition}_{name.replace(' ', '-')}.pdf')
        plt.close()


if __name__ == '__main__':
    apply_matplotlib_style()
    main()
