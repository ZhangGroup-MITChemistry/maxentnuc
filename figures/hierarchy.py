from correlation_length import load_trajectories
from maxentnuc.analysis.analysis import get_local_rg
import numpy as np
from tqdm import tqdm
from maxentnuc.analysis.domain_analyzer import DomainAnalyzer, Domains
from maxentnuc.analysis.insulation import get_insulation_scores, get_segments
from neighbor_balance.plotting import ContactMap, apply_matplotlib_style
import matplotlib.pyplot as plt
from maxentnuc.analysis.insulation import get_insulation_scores, get_segments


def get_rg_matrix(positions):
    n = positions.shape[0]
    rg_matrix = np.zeros((n, n))
    for w in range(3, n, 2):
        rg = get_local_rg(positions[None, :], window=w)
        for i in range(n - w + 1):
            # Upper triangle
            rg_matrix[i, i+w-1] = rg[i+w//2]
            rg_matrix[i, i+w-2] = rg[i+w//2]

            # Lower triangle
            rg_matrix[i+w-1, i] = rg[i+w//2]
            rg_matrix[i+w-2, i] = rg[i+w//2]
    return rg_matrix


def get_dbscan(example):
    analyzer = DomainAnalyzer(30, 200)

    domains = analyzer.analyze_frame(example)
    sections = domains.labels

    dbscan_mat = sections[:, None] == sections[None, :]
    dbscan_mat &= (sections[:, None] != -1) & (sections[None, :] != -1)
    dbscan_mat = dbscan_mat.astype(float)
    dbscan_mat[dbscan_mat == 0] = np.nan
    return domains, dbscan_mat


def get_insulation(example):
    insulation = get_insulation_scores(example[None, :], windows=[10])[10][0]
    mask = insulation > -2.5
    sections = get_segments(mask)

    domains_ins = Domains(example.copy(), sections)

    ins_mat = sections[:, None] == sections[None, :]
    ins_mat &= (sections[:, None] != -1) & (sections[None, :] != -1)
    ins_mat = ins_mat.astype(float)
    ins_mat[ins_mat == 0] = np.nan
    return domains_ins, ins_mat


def plot_rg(rg_matrix, dbscan_matrix, ins_matrix, zooms):
    f, ax = plt.subplots(1, len(zooms), figsize=(2.5*len(zooms), 2.5), gridspec_kw={'wspace': 0.2})
    for i, zoom in enumerate(zooms):
        _rg_matrix = rg_matrix[zoom[0]:zoom[1], zoom[0]:zoom[1]]**2
        if i == 0:
            _ins_matrix = dbscan_matrix[zoom[0]:zoom[1], zoom[0]:zoom[1]] * np.max(_rg_matrix)
        else:
            _ins_matrix = ins_matrix[zoom[0]:zoom[1], zoom[0]:zoom[1]] * np.max(_rg_matrix)
        mat = np.triu(_rg_matrix) + np.tril(_ins_matrix)
        cmap = ContactMap(mat, chrom='chr1', start=zoom[0]*200, end=zoom[1]*200, resolution=200)
        
        vmax = 100 * (zoom[1] - zoom[0])**(2/3)
        im = cmap.plot_contact_map(log_norm=False, cmap='viridis_r', ax=ax[i], vmin=0, vmax=vmax, colorbar=False)
        cbar = plt.colorbar(im, ax=ax[i], fraction=0.046, pad=0.04, extend='max', location='top')
        rounded = int(np.sqrt(vmax))
        cbar.set_ticks([rounded**2])
        cbar.set_ticklabels([f"${rounded}^2$"])
    
        ax[i].set_yticklabels([])
        ax[i].tick_params(axis='x', labelsize=8)

        if i < len(zooms) - 1:
            start = 200 * zooms[i+1][0]
            end = 200 * zooms[i+1][1]
            ax[i].plot([start, start, end, end, start], [start, end, end, start, start], color='magenta', linewidth=1)
    return f, ax


def main(name='fbn2', frame=-1):
    
    trajectories = load_trajectories(skip=1100)
    example = trajectories[name][frame]

    rg_matrix = get_rg_matrix(example)
    domains, dbscan_matrix = get_dbscan(example)
    domains_ins, ins_matrix = get_insulation(example)

    zooms = [(0, 5000), (200, 2700), (500, 1500), (750, 1000)]
    plot_rg(rg_matrix, dbscan_matrix, ins_matrix, zooms)

    plt.savefig(f'img/hierarchy_{name}.pdf', dpi=200)
    plt.close()

    domains.plot_all_2D()
    plt.savefig(f'img/hierarchy_{name}_render.pdf', dpi=200)
    plt.close()

if __name__ == '__main__':
    apply_matplotlib_style()
    main(name='fbn2', frame=-1)
    main(name='klf1', frame=-1)
