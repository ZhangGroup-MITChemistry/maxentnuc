from multiprocessing import Pool
from maxentnuc.analysis.analysis import *
from maxentnuc.analysis.mei_analyzer import MEIAnalyzer
from maxentnuc.analysis.insulation import get_insulation_scores, get_segments
from neighbor_balance.plotting import format_ticks, apply_matplotlib_style, stylize_gene_name
from scipy.spatial import Delaunay
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


def get_intrusions(positions, indices, plot=''):
    hull = ConvexHull(positions[indices])
    hull_delaunay = Delaunay(positions[indices][hull.vertices])
    intrusion_mask = hull_delaunay.find_simplex(positions) >= 0
    intrusion_mask[indices] = False
    intrusion_count = np.sum(intrusion_mask)

    if plot:
        fig = plt.figure(figsize=(4, 4))
        ax = fig.add_subplot(111, projection="3d")
        # ax.scatter(positions[~intrusion_mask, 0], positions[~intrusion_mask, 1], positions[~intrusion_mask, 2],
        #            c=np.arange(len(positions[~intrusion_mask])), cmap="turbo", alpha=0.3)
        
        ax.plot(positions[:, 0], positions[:, 1], positions[:, 2], c='gray')
        ax.scatter(positions[indices][:, 0], positions[indices][:, 1], positions[indices][:, 2],
                   c='green')
        ax.scatter(positions[intrusion_mask][:, 0], positions[intrusion_mask][:, 1], positions[intrusion_mask][:, 2],
                   color="black")

        # for simplex in hull.simplices:
        #     simplex = np.append(simplex, simplex[0])
        #     ax.plot(positions[indices][simplex, 0], positions[indices][simplex, 1], positions[indices][simplex, 2], "m-")

        ax.add_collection3d(Poly3DCollection(positions[indices][hull.simplices], alpha=0.2, edgecolor="none"))

        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        plt.savefig(plot)
        plt.close()

    return intrusion_count


def get_intrusions_track(positions, window=25):
    intrusions = np.full(len(positions), np.nan)
    for i in range(window, len(positions)-window):
        intrusions[i] = get_intrusions(positions, np.arange(i-window, i+window+1))
    return intrusions


def get_intrusions_matrix(positions, windows):
    intrusions = []
    for window in windows:
        intrusions.append(get_intrusions_track(positions, window))
    return np.array(intrusions)


def get_purity(mat, windows):
    size = 2 * windows + 1
    purity = size.reshape(-1, 1) / (mat + size.reshape(-1, 1))
    return size, purity


def get_max_purity(mat, windows):
    size, purity = get_purity(mat, windows)
    max_purity = np.zeros(purity.shape)
    for i, w in enumerate(windows):
        for j in range(len(mat[i])):
            max_purity[i, j] = np.nanmax(purity[i, max(0, j-w):min(purity.shape[1], j+w+1)])
    return size, max_purity


def plot_intrusion_matrix(mat, windows, skip=3, NRL=200):
    f, ax = plt.subplots(figsize=(6, 2), gridspec_kw={'wspace': 0.05})
    size, purity = get_purity(mat, windows)
    plt.imshow(purity[::-1], aspect='auto', interpolation='none', cmap='coolwarm', vmin=0.75, vmax=1,
               extent=(0, mat.shape[1]*NRL, 0, len(size)))
    
    yticks = [f'{x} KB' for x in NRL*size[::skip] / 1000]
    ax.set_yticks(np.arange(len(size))[::skip], yticks)
    print(NRL*size[::-1][::skip])
    plt.colorbar()
    format_ticks(ax, y=False)
    ax.set_xlabel('Position')
    ax.set_ylabel('Window Size')
    ax.set_title('Purity = 1 - Intrusions / Window Size')
    return f, ax


def plot_purity(size, max_purity, ax=None, NRL=200, label=None):
    """
    Example usage:
    >>> size, max_purity = get_max_purity(mat, windows)
    >>> plot_purity(size, max_purity)
    """
    if ax is None:
        f, ax = plt.subplots(figsize=(3, 2))

    x = NRL*size
    y = np.nanmean(max_purity == 1.0, axis=1)

    x = np.concatenate([[NRL], x])
    y = np.concatenate([[1], y])

    ax.plot(x, y, label=label)
    ax.set_xscale('log')
    ax.set_xlabel('Window Size (bp)')
    ax.set_ylabel('Fraction unmixed')
    format_ticks(ax, y=False)
    ax.set_ylim(0, 1)
    return ax

#######################################################################


def load_trajectories(skip=550):
    mei_runs = '/orcd/data/binz/001/joepaggi/mei_runs'
    prod = {
        'nanog': {'config': f'{mei_runs}/nanog/v36/config.yaml', 'iteration': 18},
        'klf1': {'config': f'{mei_runs}/klf1/v3/config.yaml', 'iteration': 11},
        'ppm1g': {'config': f'{mei_runs}/ppm1g/v2/config.yaml', 'iteration': 10},
        #'ppm1g_ice': {'config': f'{mei_runs}/ppm1g/v3/config.yaml', 'iteration': 12},
        #'ppm1g_rad21_aid': {'config': f'{mei_runs}/ppm1g/v4/config.yaml', 'iteration': 24},
        'sox2': {'config': f'{mei_runs}/sox2/v11/config.yaml', 'iteration': 13},
        'fbn2' : {'config': f'{mei_runs}/fbn2/v2/config.yaml', 'iteration': 12},
    }

    for name in prod:
        prod[name]['mei'] = MEIAnalyzer(prod[name]['config'], scale=0.1)

    trajectories = {}
    for name, info in prod.items():
        trajectory = info['mei'].get_positions(info['iteration'], skip=skip, burnin=0)
        trajectory = trajectory.reshape(-1, *trajectory.shape[-2:])
        trajectories[name] = trajectory
        print(name, trajectory.shape)
    return trajectories


def plot_example(example):
    get_intrusions(example.copy()[200:500], np.arange(80, 120),
                   plot='img/correlation_length_example_intrusions.pdf')
    
    get_intrusions(example.copy()[200:500], np.arange(185, 225),
                   plot='img/correlation_length_example2_intrusions.pdf')

    windows = np.unique(np.logspace(np.log10(4), np.log10(4000), 30).astype(int))
    mat = get_intrusions_matrix(example, windows=windows)
    plot_intrusion_matrix(mat, windows, skip=3)
    plt.savefig('img/correlation_length_log.pdf', dpi=200)
    plt.close()

    size, max_purity = get_max_purity(mat, windows)
    plot_purity(size, max_purity)
    plt.savefig('img/correlation_length_example.pdf')
    plt.close()

    windows = np.unique(np.linspace(5, 250, 30).astype(int))
    mat = get_intrusions_matrix(example, windows=windows)
    plot_intrusion_matrix(mat, windows, skip=3)
    plt.savefig('img/correlation_length_linear.pdf', dpi=200)
    plt.close()


def plot_aggregrate(trajectories):
    f, ax = plt.subplots(figsize=(3, 2))
    windows = np.unique(np.logspace(np.log10(4), np.log10(4000), 30).astype(int))

    for name, trajectory in trajectories.items():
        print(f'Processing {name} {len(trajectory)}', flush=True)

        purities = []
        for positions in trajectory:
            mats = get_intrusions_matrix(positions, windows=windows)
            size, max_purity = get_max_purity(mats, windows)
            purities.append(max_purity)

        plot_purity(size, np.hstack(purities), label=stylize_gene_name(name), ax=ax)
    ax.legend()
    return ax


def plot_insulated_domains(trajectories, NRL=200):
    f, ax = plt.subplots(3, figsize=(3, 4), sharex=True)
    for name, trajectory in trajectories.items():
        #trajectory = trajectory[::10]
        print(f'Processing {name} {len(trajectory)}', flush=True)
        insulations = get_insulation_scores(trajectory, windows=[10])[10]

        thresholds = [-1.5, -2, -2.5, -3, -3.5]
        mean_sizes = []
        mean_purities = []
        mean_unmixed = []
        for threshold in thresholds:
            purities, sizes, concs = [], [], []
            for frame, insulation  in zip(trajectory, insulations):
                mask = insulation > threshold
                sections = get_segments(mask)
                for clutch in range(np.max(sections)):
                    indices = np.arange(sections.shape[0])[sections == clutch]
                    size = len(indices)
                    if size < 4:
                        intrusions = 0
                    else:
                        intrusions = get_intrusions(frame, indices)
                    purity = size / (intrusions + size)
                    sizes += [size]
                    purities += [purity]
                concs += [np.mean(mask)]
            mean_sizes += [np.mean(sizes)*NRL]
            mean_purities += [np.mean(purities)]
            mean_unmixed += [np.mean(np.array(purities) == 1.0)]
    
        ax[0].plot(thresholds, mean_purities)
        ax[1].plot(thresholds, mean_unmixed, label=stylize_gene_name(name))
        ax[2].plot(thresholds, mean_sizes, label=stylize_gene_name(name))

    for a in ax:
        a.axvline(-2.5, color='red')

    ax[0].set_ylabel('Mean Purity')
    ax[0].set_ylim(0.75, 1)
    ax[0].set_ylim(0, 1)
    ax[1].set_ylabel('Fraction Unmixed')
    ax[1].set_ylim(0, 1)
    ax[2].set_xlabel('Insulation Threshold')
    ax[2].set_ylabel('Mean Size')
    ax[2].set_ylim(0)
    ax[2].legend(fontsize=6, loc='upper right')
    format_ticks(ax[2], x=False)


def main():
    apply_matplotlib_style()
    trajectories = load_trajectories(skip=110)

    plot_example(trajectories['fbn2'][0])

    plot_insulated_domains(trajectories)
    plt.savefig('img/correlation_length_insulated_domains.pdf')
    plt.close()

    plot_aggregrate(trajectories)
    plt.savefig('img/correlation_length.pdf')
    plt.close()


if __name__ == '__main__':
    main()


# This directly computes the overlap between insulated domains using convex hulls.
# import trimesh

# CONTACT_RADIUS = 40

# def convex_hull_overlap_volume(points1, points2):
#     if len(points1) < 4 or len(points2) < 4:
#         return 0, 1, 1
#     hull1 = trimesh.PointCloud(points1).convex_hull
#     hull2 = trimesh.PointCloud(points2).convex_hull
#     intersection = hull1.intersection(hull2)
#     return intersection.volume, hull1.volume, hull2.volume

# def overlap_matrix(positions, sections):
#     section_indices = np.unique(sections)
#     section_indices = section_indices[section_indices >= 0]
#     overlaps = np.zeros((len(positions), len(positions)))
#     for i in section_indices:
#         print(i)
#         for j in section_indices:
#             if j >= i:
#                 d = get_distance_matrix(positions[sections == i], positions[sections == j])
#                 if np.all(d > CONTACT_RADIUS):
#                     continue
#                 o, a, b = convex_hull_overlap_volume(positions[sections == i], positions[sections == j])
#                 mask = (sections == i)[:, None] & (sections == j)[None, :]
#                 overlaps[mask] = o / min(a, b)
#                 overlaps[mask.T] = o / min(a, b)
#                 if o / min(a, b) > 0.5 and i != j:
#                     print(i, j, o / min(a, b))
#     return overlaps

# def contact_matrix(positions, sections):
#     section_indices = np.unique(sections)
#     section_indices = section_indices[section_indices >= 0]
#     contacts = np.zeros((len(positions), len(positions)))
#     for i in section_indices:
#         print(i)
#         for j in section_indices:
#             if j >= i:
#                 d = get_distance_matrix(positions[sections == i], positions[sections == j])
#                 o = np.sum(np.any(d < CONTACT_RADIUS, axis=1))
#                 if o == 0:
#                     continue
#                 a = np.sum(sections == i)
#                 b = np.sum(sections == j)
#                 mask = (sections == i)[:, None] & (sections == j)[None, :]
#                 contacts[mask] = o / min(a, b)
#                 contacts[mask.T] = o / min(a, b)
#     return contacts

# def plot_overlaps(domain_ins, domain_dbscan):
#     overlaps = overlap_matrix(domain_ins.positions, domain_ins.labels)
#     contacts = contact_matrix(domain_ins.positions, domain_ins.labels)

#     f, ax = plt.subplots(1, 3, figsize=(19, 5))

#     ax[0].set_title('Overlaps')
#     im = ax[0].imshow(overlaps, interpolation='none', vmin=0, vmax=1, cmap='fall')
#     plt.colorbar(im)

#     ax[1].set_title('Contacts')
#     im = ax[1].imshow(contacts, interpolation='none', vmin=0, vmax=1, cmap='fall')
#     plt.colorbar(im)

#     ax[2].set_title('DBSCAN')
#     color = np.stack([domain_dbscan.labels]*domain_dbscan.positions.shape[0], axis=0)
#     same = domain_dbscan.labels[:, None] == domain_dbscan.labels[None, :]
#     color[~same] = -1
#     im = ax[2].imshow(color, cmap=domain_dbscan.colors(noise_color='white'), interpolation='none')
#     plt.colorbar(im)
    

# #domain_dbscan = domains['nanog'][-1]
# #domain_ins = get_domain_ins(domain_dbscan.positions, window=10, threshold=median_insulation[10]-1, use_peaks=False)
# plot_overlaps(d, dbscan)
# plt.savefig('insulation_overlaps.pdf')
# plt.show()
