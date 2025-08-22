from neighbor_balance.plotting import ContactMap, get_distance_average, apply_matplotlib_style
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

data_root = '/home/joepaggi/share/rcmc/WT_BR1'

def plot_base_ps():
    df = pd.read_csv(f'{data_root}/select.nodups.pairs.gz.base_ps.csv')
    nrl = 190
    f, ax = plt.subplots(1, 2, figsize=(6, 2), sharey=True, sharex=True)
    x = df['distance'].to_numpy()
    for color, direction in zip(['cornflowerblue', 'mediumvioletred', 'darkorange', 'gold'], ['inward', 'outward', 'tandem_entry', 'tandem_exit']):
        ax[0].plot(df['distance'], df[direction], label=direction, color=color, ls = '--' if direction == 'tandem_exit' else '-')

        if direction == 'inward':
            shift = -130
        elif direction == 'outward':
            shift = 130
        else:
            shift = 0
        ax[1].plot(df['distance'] + shift, df[direction], label=direction, color=color, ls = '--' if direction == 'tandem_exit' else '-')

    for i in range(2):
        for j in range(1, 11, 2):
            ax[i].axvspan(j*nrl - nrl // 2, j*nrl + nrl // 2, color='gray', alpha=0.2)

        ax[i].set_xticks(np.arange(nrl, 1501, nrl))
        ax[i].set_xticklabels([f'i+{j}' for j in range(1, 8)])
        ax[i].set_xlim(0, 1500)
        ax[i].set_ylim(0)

        ax[i].set_xlabel('Distance (bp)')
    ax[0].set_ylabel('Read count')
    ax[0].legend()
    ax[0].set_title('Raw Reads')
    ax[1].set_title('Shifted Reads')

def plot_contact_map_ps():
    regions = {
        'klf1': 'chr8:84,900,000-85,800,000',
        'ppm1g': 'chr5:31,300,000-32,300,000',
        'nanog': 'chr6:122,480,000-122,850,000',
        'fbn2': 'chr18:58,042,000-59,024,000',
        'sox2': 'chr3:33,800,000-35,700,000',
    }


    cools = {
        'raw': f'{data_root}/select.mcool',
        'shifted': f'{data_root}/select_corrected.mcool',
        'shifted minus inward': f'{data_root}/select_corrected_minus_inward.mcool'
    }
    f, ax = plt.subplots(figsize=(3, 2))
    for name, path in cools.items():
        print(f'Processing {name}...')
        ps = np.zeros(8)
        for region_name, region in regions.items():
            contact_map = ContactMap.from_cooler(path, 200, region)
            contact_map.contact_map /= np.nanmean(np.diagonal(contact_map.contact_map, 1))
            ps += get_distance_average(contact_map.contact_map, show_self=True)[:len(ps)]
        ps *= 0.5 / ps[-1]
        plt.plot(np.arange(8), ps, label=name, marker='o')
    plt.xticks(np.arange(1, 8), [f'i+{i}' for i in range(1, 8)])
    #plt.ylim(0, 1.1)
    plt.xlim(0, 7)
    plt.xlabel('Distance (nrl)')
    plt.ylabel('Contact frequency')
    plt.legend()

if __name__ == '__main__':
    apply_matplotlib_style()
    plot_base_ps()
    plt.savefig(f'img/base_ps.pdf')
    plt.close()
    plot_contact_map_ps()
    plt.savefig(f'img/contact_map_ps.pdf')
    plt.close()
