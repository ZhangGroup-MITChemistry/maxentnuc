from scipy.io import loadmat
from .track_data import *
from maxentnuc.simulation.experimental_contacts import ContactMap
from neighbor_balance.neighbor import normalize_contact_map_average, normalize_contact_map_neighbor
from neighbor_balance.plotting import ContactMap, parse_region, format_ticks
from .analysis import *
from matplotlib.pyplot import cm
from scipy.stats import pearsonr


def nanog_index():
    n = 32
    index_imaged = np.zeros(n, dtype=bool)
    for i in range(0, 9, 3):
        index_imaged[i] = True
    for i in range(9, 9 + 13):
        index_imaged[i] = True
    for i in range(9 + 13 + 2, 2 * 9 + 13, 3):
        index_imaged[i] = True
    index_imaged[-1] = True

    index_5kb = np.zeros(n, dtype=bool)
    for i in range(9, 9 + 13):
        index_5kb[i] = True

    index_15kb = np.zeros(n, dtype=bool)
    for i in range(0, n, 3):
        index_15kb[i] = True

    return index_imaged, index_5kb, index_15kb


def read_nanog_orca(data_root='/home/gridsan/jpaggi'):
    chrom = 'chr6'
    tss = 122667815 + 15000, 122672815 + 15000

    orca_fnames = {
        'rep1': f'{data_root}/nanog/mESdiffn_NanogORCA_rep1.mat',
        'rep2': f'{data_root}/nanog/mESdiffn_NanogORCA_rep2.mat'
    }

    regions = []
    for i in range(-3 * 3 - 6, 3 * 3 + 6 + 1):
        s = tss[0] + i * 5000
        regions.append((s, s + 5000))
    regions += [regions[3 * 3 + 3]]

    index_imaged, index_5kb, index_15kb = nanog_index()

    orca = {}
    for rep, fname in orca_fnames.items():
        orca[rep] = {}
        for index, name in enumerate(['esc', 'day1', 'day2', 'day3']):
            _positions = loadmat(fname)['dat'][0, index][0].transpose(2, 0, 1)
            positions = np.zeros((_positions.shape[0], len(regions), 3))
            positions[:] = np.nan
            positions[:, index_imaged] = _positions
            orca[rep][name] = TrackData(chrom, regions, positions)
    return orca


def read_sox2_orca(wt_only=False, data_root='/home/gridsan/jpaggi'):
    # Add the below number to get from mm10 to mm39 for the sox2 locus.
    # mm39: chr3:33,814,000-35,694,000 -> mm10: chr3:33,759,851-35,639,851
    MM10_TO_MM39 = 33_814_000 - 33_759_851

    orca = {'wt': '4DNFIA63SK83',
            'ctcf_nt_1:': '4DNFI2OG3IPB',
            'ctcf_nt_2:': '4DNFIU5X9MXT',
            'ctcf_nt_3:': '4DNFIIQ8YIG4',
            'rad21_nt_1:': '4DNFI4SYERQD',
            'rad21_nt_2:': '4DNFILHN6ZHZ',
            'rad21_nt_3:': '4DNFIPUTK6GO',
            'rad21_aux_1': '4DNFI1MBPSTC',
            'rad21_aux_2': '4DNFIHA5PA9U',
            'rad21_aux_3': '4DNFIDNY8W7L',
            # 'huang_1': '4DNFIKPGMZJ8'
            }

    if wt_only:
        orca = {'wt': orca['wt']}

    orca = {k: TrackData.from_trace_core(f'{data_root}/orca/{v}.csv') for k, v in orca.items()}
    for name, track in orca.items():
        track.shift_coordinates(MM10_TO_MM39)

    for name, track in orca.items():
        assert track.is_compact()
    return orca


def load_simulations(coarse_regions, conditions, chrom, start, end, skip=100, burnin=0):
    simulation = {}
    for name, info in conditions.items():
        print(name)
        positions = load_polymer(info['psf'], info['dcd'], info['selection'], info['scale'], skip=skip, burnin=burnin)
        positions = np.array(positions)
        regions = [(s, s + 200) for s in range(start, end, 200)]
        simulation[name] = TrackData(chrom, regions, positions)
        simulation[name] = simulation[name].coarsen(coarse_regions)
    return simulation


from neighbor_balance.plotting import ContactMap
from neighbor_balance.smoothing import interpolate_diagonals

def make_cg_contact_map(contact_map, regions):
    for i in range(len(regions)-1):
        assert regions[i][1] <= regions[i+1][0], f"Regions {regions[i]} and {regions[i+1]} are not in order"

    for region in regions:
        assert region[0] % contact_map.resolution == 0
        assert region[1] % contact_map.resolution == 0
        assert region[0] >= contact_map.start
        assert region[1] <= contact_map.end
        assert region[1] - region[0] == regions[0][1] - regions[0][0]

    cg_contact_map = np.zeros((len(regions), len(regions)))
    for i in range(len(regions)):
        for j in range(len(regions)):
            start_i = (regions[i][0] - contact_map.start) // contact_map.resolution
            end_i = (regions[i][1] - contact_map.start) // contact_map.resolution + 1
            start_j = (regions[j][0] - contact_map.start) // contact_map.resolution
            end_j = (regions[j][1] - contact_map.start) // contact_map.resolution + 1
            cg_contact_map[i, j] = np.nanmean(contact_map.contact_map[start_i:end_i, start_j:end_j])

    start_i = (regions[0][0] - contact_map.start) // contact_map.resolution
    end_i = (regions[-1][1] - contact_map.start) // contact_map.resolution
    start = contact_map.start + start_i * contact_map.resolution
    end = contact_map.start + end_i * contact_map.resolution
    resolution = (end - start) // len(regions)
    return ContactMap(cg_contact_map, contact_map.chrom, start, end, resolution)

def load_contact_maps(contact_map_npz, regions):
    ice_contact_map = ContactMap.from_npz(contact_map_npz)
    ice_contact_map.contact_map = normalize_contact_map_average(ice_contact_map.contact_map)
    ice_contact_map = make_cg_contact_map(ice_contact_map, regions)
    ice_contact_map.contact_map = normalize_contact_map_average(ice_contact_map.contact_map)

    neighbor_contact_map = ContactMap.from_npz(contact_map_npz)
    neighbor_contact_map.contact_map = normalize_contact_map_neighbor(neighbor_contact_map.contact_map)
    neighbor_contact_map = make_cg_contact_map(neighbor_contact_map, regions)
    neighbor_contact_map.contact_map = normalize_contact_map_average(neighbor_contact_map.contact_map)
    return ice_contact_map, neighbor_contact_map

########################################################################################################################


def power_law(x, a, b):
    return a*x**b


def kernel_smooth(x, y, bw):
    d = np.abs(x.reshape(-1, 1) - x.reshape(1, -1))
    w = np.exp(-d**2 / (2*bw**2))
    w /= w.sum(axis=1)
    return np.sum(w * y, axis=1)


def get_section_distances(positions):
    dists = positions_to_average_distance(positions)
    section_dists = []
    for i in range(dists.shape[0]):
        section_dists += [np.diagonal(dists, i).mean()]
    return np.array(section_dists)


def plot_distance_scaling(ax, name, track, cut=20, c=None, ls=None, distance_scale=1, scatter=True):
    x = track.x()
    x -= x[0]
    dists = distance_scale * get_section_distances(track.positions)

    popt, _ = curve_fit(power_law, x[cut:], dists[cut:])
    a, b = popt
    for j in range(2):
        ax[j].plot(x, dists, label=name, color=c, ls=ls)
        if scatter:
            ax[j].scatter(x, dists, color=c)

    ax[0].set_ylabel('Distance (nm)')
    ax[1].set_yscale('log')
    ax[1].set_xscale('log')
    format_ticks(ax[0], y=False)
    format_ticks(ax[1], y=False)
    return ax

########################################################################################################################


def compare(X, Y, ax=None, plot=True):
    n = X.shape[0]
    colors = cm.rainbow(np.linspace(0, 1, n))
    corr = []

    for offset, color in zip(range(1, n), colors):
        x, y = np.diagonal(X, offset), np.diagonal(Y, offset)
        if plot:
            ax.scatter(x, y, label=f'offset={offset}', color=color)
        if len(x) > 3:
            corr += [pearsonr(x, y)[0]]
    return corr, pearsonr(full_to_triu(X, k=1), full_to_triu(Y, k=1))


def scatter_grid(measurements):
    f, ax = plt.subplots(len(measurements)-1, len(measurements)-1, figsize=(16, 16))
    for i in range(len(measurements)-1):
        for j in range(1, len(measurements)):
            if i >= j:
                ax[i, j-1].axis('off')
                continue
            print(measurements[i][0], measurements[j][0])
            offset_corr, global_corr = compare(measurements[j][1], measurements[i][1], ax=ax[i, j-1])
            text = f'{global_corr[0]:.2f}\n{np.mean(offset_corr):.2f}'
            ax[i, j-1].text(0.05, 0.95, text, transform=ax[i, j-1].transAxes, va='top', ha='left')
            if i < j-1:
                ax[i, j-1].set_xticklabels([])
                ax[i, j-1].set_yticklabels([])

    for i in range(1, len(measurements)):
        ax[0, i-1].set_title(measurements[i][0], size=16, rotation=45, ha='left')
    for j in range(len(measurements)-1):
        ax[j, j].set_ylabel(measurements[j][0], size=16, rotation=0, ha='right')

########################################################################################################################


def hist_grid(sim, sim_noisy, orca_rep1, orca_rep2=None, high=800):
    n = sim.shape[1]

    bins = np.linspace(0, high, 40)
    f, ax = plt.subplots(n-1, n-1, figsize=(25, 25))
    for i in range(n-1):
        for j in range(1, n):
            if i >= j:
                ax[i, j-1].axis('off')
                continue
            ax[i, j-1].hist(orca_rep1[:, i, j], bins=bins, density=True, alpha=0.5, color='green')
            if orca_rep2 is not None:
                ax[i, j-1].hist(orca_rep2[:, i, j], bins=bins, density=True, alpha=0.5, color='orange')
            ax[i, j-1].hist(sim[:, i, j], bins=bins, density=True, histtype='step', color='k')
            ax[i, j-1].hist(sim_noisy[:, i, j], bins=bins, density=True, histtype='step', color='blue')
            ax[i, j-1].set_yticks([])
            ax[i, j-1].set_xlim(0, high)
            ax[i, j-1].set_xticks(range(0, high, 200))
            for d in range(0, high, 200):
                ax[i, j-1].axvline(d, color='gray', linestyle='--', alpha=0.5)

    for i in range(n):
        ax[0, i-1].set_title(i-n//2, size=16)

    for i in range(n-1):
        ax[i, i+1-1].set_ylabel(i-n//2, rotation=0, size=16, ha='right')


def hist_grid_examples(start, sim, sim_noisy, orca_rep1, orca_rep2=None, high=800, rows=5, cols=4, stride_kb=30):
    bins = np.linspace(0, high, 40)
    f, axs = plt.subplots(rows, cols, figsize=(10, 10), sharex=True)
    for spacing_i, spacing in enumerate(np.linspace(1, sim.shape[1] - rows, cols).astype(int)):
        print(spacing_i, spacing, sim.shape[1] - rows, (sim.shape[1] - rows) // cols)
        ax = axs[0, spacing_i]
        ax.text(0.5, 1.15, f'{spacing * stride_kb} kb', fontsize=16, ha="center", transform=ax.transAxes)

        for example_i, example in enumerate(np.linspace(0, sim.shape[1] - spacing - 1, rows).astype(int)):
            ax = axs[example_i, spacing_i]

            i = example
            j = example + spacing

            ax.set_title(f'{start + i * stride_kb * 1000:,} to {start + j * stride_kb * 1000:,}', size=8, pad=1)

            ax.hist(orca_rep1[:, i, j], bins=bins, density=True, alpha=0.5, color='green', label='ORCA')
            if orca_rep2 is not None:
                ax.hist(orca_rep2[:, i, j], bins=bins, density=True, alpha=0.5, color='orange', label='ORCA2')

            ax.hist(sim[:, i, j], bins=bins, density=True, histtype='step', color='k', label='Simulation')
            ax.hist(sim_noisy[:, i, j], bins=bins, density=True, histtype='step', color='blue',
                    label='Simulation + noise')

            ax.set_yticks([])
            ax.set_xlim(0, high)
            ax.set_xticks(range(0, high, 200))
            for d in range(0, high, 200):
                ax.axvline(d, color='gray', linestyle='--', alpha=0.5)

    axs[0, -1].legend(fontsize=8)

    for ax in axs[-1, :]:
        ax.set_xticklabels(ax.get_xticks(), rotation=45)

########################################################################################################################

def plot_pairwise(data, region, **kwargs):
    chrom, start, end = parse_region(region)
    resolution = (end - start) // len(data)
    ContactMap(data, chrom, start, end, resolution).plot_contact_map(**kwargs)

def compare_contacts(orca, simulation, simulation_noisy, contact_map, region, C):
    orca_contacts = positions_to_contacts(orca.positions, C)

    rcmc_contacts = contact_map.copy()
    sim_contacts = positions_to_contacts(simulation.positions, C)
    sim_noisy_contacts = positions_to_contacts(simulation_noisy.positions, C)

    np.fill_diagonal(orca_contacts, np.nan)
    np.fill_diagonal(rcmc_contacts, np.nan)
    np.fill_diagonal(sim_contacts, np.nan)

    f, ax = plt.subplots(3, 4, figsize=(25, 15))

    cmap = 'fall'
    vmin = 1e-3
    vmax = 1
    ax[0, 0].set_title('RCMC')
    plot_pairwise(rcmc_contacts, region, ax=ax[0, 0], vmax=vmax, vmin=vmin)
    ax[0, 1].set_title('Simulation')
    plot_pairwise(sim_contacts, region, ax=ax[0, 1], vmax=vmax, vmin=vmin)
    ax[0, 2].set_title('Simulation + noise')
    plot_pairwise(sim_noisy_contacts, region, ax=ax[0, 2], vmax=vmax, vmin=vmin, cmap=cmap)
    ax[0, 3].set_title('ORCA')
    plot_pairwise(orca_contacts, region, ax=ax[0, 3], vmax=vmax, vmin=vmin, cmap=cmap)

    cmap = 'coolwarm'
    vmin = -1
    vmax = 1
    ax[1, 0].set_title('log2(RCMC / ORCA)')
    plot_pairwise(np.log2(rcmc_contacts / orca_contacts), region, ax=ax[1, 0], vmax=vmax, vmin=vmin, cmap=cmap, log_norm=False)
    ax[1, 1].set_title('log2(Simulation / ORCA)')
    plot_pairwise(np.log2(sim_contacts / orca_contacts), region, ax=ax[1, 1], vmax=vmax, vmin=vmin, cmap=cmap, log_norm=False)
    ax[1, 2].set_title('log2(Simulation + noise / ORCA)')
    plot_pairwise(np.log2(sim_noisy_contacts / orca_contacts), region, ax=ax[1, 2], vmax=vmax, vmin=vmin, cmap=cmap, log_norm=False)
    ax[1, 3].axis('off')

    def scatter(ax, x, y):
        _x = full_to_triu(x, k=1)
        _y = full_to_triu(y, k=1)
        ax.scatter(_x, _y, c=(_y-_x)/_x, cmap=cmap, vmin=vmin, vmax=vmax, s=3)

        _x = [np.diagonal(x, i).mean() for i in range(1, x.shape[0])]
        _y = [np.diagonal(y, i).mean() for i in range(1, y.shape[0])]
        ax.plot(_x, _y, color='black')

        ax.plot([0, .2], [0, .2], color='grey', linestyle='--')
        ax.set_xlabel('ORCA')
        ax.set_yscale('log')
        ax.set_xscale('log')

    ax[2, 0].set_ylabel('RCMC')
    scatter(ax[2, 0], orca_contacts, rcmc_contacts)
    ax[2, 1].set_ylabel('Sim')
    scatter(ax[2, 1], orca_contacts, sim_contacts)
    ax[2, 2].set_ylabel('Sim + noise')
    scatter(ax[2, 2], orca_contacts, sim_noisy_contacts)
    ax[2, 3].axis('off')

def compare_dists(orca, simulation, simulation_noisy, region, max_dist=600):
    orca_dists = positions_to_average_distance(orca.positions)
    sim_dists = positions_to_average_distance(simulation.positions)
    sim_noisy_dists = positions_to_average_distance(simulation_noisy.positions)

    f, ax = plt.subplots(1, 4, figsize=(15, 3), sharey=True)

    cmap = 'viridis'
    vmin = 0
    vmax = max_dist
    ax[0].set_title('Simulation')
    plot_pairwise(sim_dists, region, ax=ax[0], vmax=vmax, vmin=vmin, cmap=cmap, log_norm=False)
    ax[1].set_title('Simulation + noise')
    plot_pairwise(sim_noisy_dists, region, ax=ax[1], vmax=vmax, vmin=vmin, cmap=cmap, log_norm=False)
    ax[2].set_title('ORCA')
    plot_pairwise(orca_dists, region, ax=ax[2], vmax=vmax, vmin=vmin, cmap=cmap, log_norm=False)

    cmap = 'coolwarm'
    vmin = -1
    vmax = 1
    ax[3].set_title('log2(Simulation + noise / ORCA)')
    plot_pairwise(np.log2(sim_noisy_dists / orca_dists), region, ax=ax[3], vmax=vmax, vmin=vmin, cmap=cmap, log_norm=False)

########################################################################################################################


class NoiseDistribution:
    def __init__(self, dist='t', scale=26, shape=1.8, max_val=float('inf')):
        self.dist = dist
        self.scale = scale
        self.shape = shape
        self.max_val = max_val

        # These are set in fit_noise.
        self.shape_space = None
        self.scale_space = None
        self.metric = None
        self.eps = None
        self.bins = None
        self.all_lls = None
        self.empirical = None
        self.optimal = None

    def __repr__(self):
        return f'NoiseDistribution(dist={self.dist}, scale={self.scale}, shape={self.shape}, max_val={self.max_val})'

    def add_noise(self, data, samples):
        data = np.repeat(data, samples, axis=0)
        if self.dist == 't':
            noise = np.random.standard_t(self.shape, size=data.shape)
        elif self.dist == 'pareto':
            noise = np.random.pareto(self.shape, size=data.shape)
            noise *= np.random.choice([-1, 1], size=data.shape)
        elif self.dist == 'normal':
            noise = np.random.normal(0, 1, size=data.shape)
        else:
            assert False
        noise *= self.scale
        noise[np.linalg.norm(noise, axis=-1) > self.max_val] = np.nan
        return data + noise

    def loss(self, p0, p1):
        if self.metric == 'll':
            return -np.sum(p0*np.log(p1))
        elif self.metric == 'kl':
            return np.sum(p0*np.log(p0/p1))
        else:
            assert False

    def histogram(self, data):
        hist, _ = np.histogram(data, bins=self.bins, density=True)
        hist = np.clip(hist, self.eps, float('inf'))
        return hist

    def fit_noise(self, orca_positions, bins, sim_positions=None, dist='t', metric='ll', spacing=1, eps=1e-10,
                  shape_space=(1, 3, 21), scale_space=(5, 40, 36), samples=100_000):
        """
        Fit the noise distribution in chromatin tracing data.

        There are two ways to fit the noise distribution depending on the available data:
        1. If a control reimaging of a locus is performed, the noise distribution can be fit to the empirical distribution of distances between imagings.
        2. Otherwise, noise distribution can be fit to correct the distances between nearby positions in a simulation to match the empirical distribution
        of corresponding distances in the imaging data.append

        Naturally, the first option is preferred, but in the absence of a control reimaging, the second option can be used to assess if the shape of
        the distace scaling in the simulations is reasonable.

        To use the first option, set `sim_positions` to None. In this case `orca_positions` should be an (M, 2, k) array where M is the number of samples
        and k is the number of dimensions. The data should be paired such that the two positons in each row are from the same sample.

        To use the second option, set both `sim_positions` and `orca_positions` to an (M, N, k) array, where N is the number of monomers in the system.
        Note that the number of samples M is not required to be the same for both datasets.

        Parameters
        ----------
        orca_positions : np.ndarray
        bins : np.ndarray
            The bins to use for the histogram of distances.
        sim_positions : np.ndarray
            The positions to use for the simulation. If None, the empirical distribution of distances in `orca_positions` is used.
        dist : str
            The distribution to use for the noise. Can be 't', 'pareto' or 'normal'.
        metric : str
            The metric to use for the loss function. Can be 'll' or 'kl'.
        spacing : int
            The spacing to use for the distances. This is the number of positions to skip when calculating the distance.
        shape_space : tuple
            The range of shape parameters to use for the noise distribution. The first element is the minimum value, the second element is the maximum value,
            and the third element is the number of values to use in the range.
        scale_space : tuple
            The range of scale parameters to use for the noise distribution. The first element is the minimum value, the second element is the maximum value,
            and the third element is the number of values to use in the range.
        samples : int
            The number of samples to use for the noise distribution. This is the total number of samples, so if there are 1,000 simulation frames, then
            samples / 1,000 noise points will be generated for each frame.
        max_val : float
            The maximum value for the noise distribution. This is used to clip the noise distribution to avoid extreme values.
        Returns
        -------
        best_params : tuple
            The best parameters for the noise distribution. This is a tuple of the form (scale, shape).
        """
        self.shape_space = np.linspace(*shape_space)
        self.scale_space = np.linspace(*scale_space)
        self.metric = metric
        self.eps = eps
        self.bins = bins
        self.dist = dist

        if sim_positions is None:
            # Use control data.
            assert orca_positions.ndim == 3 and orca_positions.shape[1] == 2
            assert spacing == 1
            sim_positions = np.zeros((1, 2, orca_positions.shape[2]))
        else:
            # Match short distances in simulation to empirical distribution.
            assert orca_positions.ndim == 3
            assert sim_positions.ndim == 3
            assert orca_positions.shape[1] == sim_positions.shape[1]

        self.empirical = np.linalg.norm(orca_positions[:, spacing:] - orca_positions[:, :-spacing], axis=-1)
        p0 = self.histogram(self.empirical)

        self.all_lls = []
        best = float('inf')
        best_params = None
        for shape in self.shape_space:
            self.all_lls += [[]]
            for scale in self.scale_space:
                self.shape, self.scale = shape, scale
                noise = self.add_noise(sim_positions, samples=samples // sim_positions.shape[0])
                noise = np.linalg.norm(noise[:, spacing:] - noise[:, :-spacing], axis=-1)
                p1 = self.histogram(noise)
                ll = self.loss(p0, p1)
                self.all_lls[-1] += [ll]
                if ll < best:
                    best = ll
                    best_params = scale, shape

        # Set parameters to the best fit.
        self.scale, self.shape = best_params
        noise = self.add_noise(sim_positions, samples=samples // sim_positions.shape[0])
        self.optimal = np.linalg.norm(noise[:, spacing:] - noise[:, :-spacing], axis=-1)
        return best_params

    def plot_noise_fit(self):
        f, ax = plt.subplots(1, 2, figsize=(10, 4))
    
        for shape, lls in zip(self.shape_space, self.all_lls):
            ax[0].plot(self.scale_space, lls, label=f'shape={shape:.2f}')

        ax[0].legend(ncol=3)
        ax[0].set_xlabel('Scale (nm)')
        ax[0].set_ylabel('Neg. log likelihood')

        self.plot_best_fit(ax=ax[1])
        return ax


    def plot_best_fit(self, ax=None):
        if ax is None:
            f, ax = plt.subplots(1, 1, figsize=(4, 3))

        x = (self.bins[1:] + self.bins[:-1]) / 2  # np.histogram uses `bins` as the bin edges, so we need to use the bin centers.

        ax.plot(x, self.histogram(self.empirical), label='Empirical', c='k')
        ax.plot(x, self.histogram(self.optimal), label=f'Fit: scale={self.scale:.1f}, shape={self.shape:.1f}', c='r', ls='--')

        textstr = '\n'.join((
        f'Median: Emp. = {np.nanmedian(self.empirical):.1f}\nFit = {np.nanmedian(self.optimal):.1f}',
        f'Variance: Emp. = {np.nanvar(self.empirical):.1f}\nFit = {np.nanvar(self.optimal):.1f}',
        f'Mean: Emp. = {np.nanmean(self.empirical):.1f}\nFit = {np.nanmean(self.optimal):.1f}',
        f'Max: Emp. = {np.nanmax(self.empirical):.1f}\nFit = {np.nanmax(self.optimal):.1f}'
        ))
        ax.text(0.98, 0.75, textstr, transform=ax.transAxes,
                verticalalignment='top', horizontalalignment='right')
        ax.legend(loc='upper right')
        ax.set_ylim(0)
        ax.set_xlim(0)
        ax.set_xlabel('Distance (nm)')
        ax.set_ylabel('Frequency (A.U.)')
        return ax


from neighbor_balance import gtf

def load_annotations(annotations, alias):
    annotations = gtf.dataframe(annotations)
    alias = pd.read_csv(alias, sep='\t')
    alias.rename(columns={'# ucsc': 'ucsc'}, inplace=True)
    annotations['chrom'] = annotations['seqname'].apply(lambda x: alias.loc[alias['refseq'] == x, 'ucsc'].values[0])
    annotations['start'] = annotations['start'].astype(int)
    annotations['end'] = annotations['end'].astype(int)
    return annotations
