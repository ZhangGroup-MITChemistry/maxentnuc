from MDAnalysis import Universe
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.spatial import Voronoi
from scipy.spatial import ConvexHull
from maxentnuc.simulation.upper_triangular import full_to_triu
from MDAnalysis.analysis.align import rotation_matrix
from scipy.optimize import curve_fit
from scipy.spatial import distance_matrix, cKDTree
from glob import glob
import logging
import matplotlib as mpl


def apply_matplotlib_style():
    logging.basicConfig(level=logging.CRITICAL)
    logging.getLogger("fontTools.subset").setLevel(logging.CRITICAL)
    mpl.rcParams['pdf.fonttype'] = 42
    mpl.rcParams['ps.fonttype'] = 42
    mpl.rcParams['savefig.transparent'] = True
    mpl.rcParams['savefig.bbox'] = 'tight'


def expand_conditions(conditions):
    new_conditions = {}
    for name, info in conditions.items():
        if 'base' in info:
            assert 'dcd' not in info
            assert 'psf' not in info
            new_conditions[name] = info.copy()
            new_conditions[name]['dcd'] = sorted(glob(info['base'] + '_trajectory.*.dcd'))
            new_conditions[name]['psf'] = info['base'] + '_topology.0.psf'
        else:
            new_conditions[name] = info.copy()

        if 'scale' not in new_conditions[name]:
            new_conditions[name]['scale'] = 2.1
        if 'selection' not in new_conditions[name]:
            new_conditions[name]['selection'] = 'name NUC'
    return new_conditions


def load_polymer(psf, dcds, selection, scale, burnin=0.5, skip=10, pad=None):
    if not isinstance(dcds, list):
        dcds = [dcds]

    positions = []
    for dcd in dcds:
        u = Universe(psf, dcd)
        start = int(len(u.trajectory) * burnin)
        _positions = u.trajectory.timeseries(u.select_atoms(selection), start=start, step=skip, order='fac')
        positions += [_positions]
    positions = np.vstack(positions)
    positions *= scale
    if pad is not None:
        positions = positions[:, pad:-pad]
    return positions


######################################################
# Utilities.


def myviolinplot(x, y, c='r', **violin_kwargs):
    v = plt.violinplot(y, x, **violin_kwargs)
    for b in v['bodies']:
        # get the center
        m = np.mean(b.get_paths()[0].vertices[:, 0])
        # modify the paths to not go further right than the center
        b.get_paths()[0].vertices[:, 0] = np.clip(b.get_paths()[0].vertices[:, 0], m, np.inf)
        b.set_color(c)


def plot_all_violins(conditions, get_metric, pad=50):
    metric = []
    for name, info in conditions.items():
        polymer_positions = load_polymer(info['psf'], info['dcd'], info['selection'], scale=info['scale'], pad=pad)
        metric += [get_metric(polymer_positions)]
    plt.figure(figsize=(10, 5))
    myviolinplot(range(len(metric)), metric)
    plt.xticks(range(len(conditions)), conditions, rotation=45, ha='right')


def plot_end_to_end_distances(conditions, separations, res=1, pad=1, kb_per_bead=0.2):
    for name, info in conditions.items():
        positions = load_polymer(info['psf'], info['dcd'], info['selection'], scale=info['scale'], pad=pad)
        positions = cg_trajectory(positions, res=res)
        y = []
        for separation in separations:
            distances = get_distances(positions, separation)
            y += [np.mean(distances)]
        x = kb_per_bead * res * separations
        plt.plot(x, y, label=name)
        plt.scatter(x, y)
    plt.xlabel('Genomic Separation (KB)')
    plt.legend()


def get_distance_matrix(points, points2=None):
    if points2 is None:
        points2 = points
    return np.linalg.norm(np.expand_dims(points, -2) - np.expand_dims(points2, -3), axis=-1)


def power_law(x, a, b):
    return a*x**b


def to_micromolar(count, volume):
    conc = count / volume  # count / nm^3
    conc /= 6.022 * 10**23  # moles / nm^3
    conc *= 1.0 * 10**24  # moles / dm^3
    conc *= 10**6  # µM
    return conc


def remove_center_of_mass_motion(trajectory, remove_rotation=False):
    """
    Remove the center of mass motion from a set of positions, and optionally remove rotations about the center of mass.
    """
    trajectory = trajectory - np.mean(trajectory, axis=-2, keepdims=True)
    if remove_rotation:
        for t in range(1, trajectory.shape[0]):
            v, _ = rotation_matrix(trajectory[t], trajectory[t-1])
            trajectory[t] = np.dot(trajectory[t], v.T)
    return trajectory


def cg_trajectory(trajectory, res=1):
    assert trajectory.shape[1] % res == 0
    n = trajectory.shape[1] // res
    cg = np.full((trajectory.shape[0], n, 3), np.nan)
    for i in range(n):
        cg[:, i] = trajectory[:, i * res:(i + 1) * res].mean(axis=1)
    return cg

###################################################
# Fluctuations.


def count_neighbors(trajectory, radius=40, probe_pad=200, probe_stride=100):
    """
    Count the number of neighbors within a distance `radius` of each position in the trajectory.

    Parameters
    ----------
    trajectory : np.ndarray
        The coordinates of the particles. Shape: (n_frames, n_particles, 3)
    radius : float
        The distance threshold for counting neighbors in nanometers.
    probe_pad : int
        The number of particles to ignore on either end of the system to avoid edge effects.
    probe_stride : int
        Sample every `probe_stride` particles to speed up the computation.

    Returns
    -------
    np.ndarray
        Shape: (n_frames, n sampled particles)
    """
    neighbors = []
    for probe in range(probe_pad, trajectory.shape[1] - probe_pad, probe_stride):
        dists = np.linalg.norm(trajectory - trajectory[:, probe:probe + 1], axis=-1)
        neighbors += [np.sum(dists < radius, axis=-1) - 1]
    return np.array(neighbors)


def plot_concentrations(conditions, cuts=(25, 30, 40, 50, 60), max_conc=700):
    f, ax = plt.subplots(1, len(cuts), figsize=(4*len(cuts), 3))
    means = np.zeros((len(conditions), len(cuts)))
    names = []
    for j, (name, info) in enumerate(conditions.items()):
        names += [name]
        trajectory = load_polymer(info['psf'], info['dcd'], info['selection'], skip=100, scale=info['scale'], pad=100, burnin=0.0)
        for i, cut in enumerate(cuts):
            neighbors = count_neighbors(trajectory, radius=cut).flatten()
            conc = to_micromolar(neighbors + 1, 4/3 * np.pi * cut**3)
            means[j, i] = np.mean(conc)

            bins = np.linspace(0, max_conc, max_conc//50)
            ax[i].hist(conc, histtype='step', label=name, bins=bins, density=True)

    ax[-1].legend(loc='center left', bbox_to_anchor=(1, 0.5))
    for i, cut in enumerate(cuts):
        ax[i].set_title(f'{cut} nm')
        ax[i].set_yticklabels([])
    plt.show()

    plt.plot(cuts, means.T, label=names)
    plt.legend()
    plt.show()


def plot_neighbors(conditions, cuts=(15, 20, 25, 30, 40, 50)):
    f, ax = plt.subplots(1, len(cuts), figsize=(4 * len(cuts), 3))
    means = np.zeros((len(conditions), len(cuts)))
    names = []
    for j, (name, info) in enumerate(conditions.items()):
        names += [name]
        trajectory = load_polymer(info['psf'], info['dcd'], info['selection'], skip=10, scale=info['scale'], pad=100,
                                 burnin=0.0)
        for i, cut in enumerate(cuts):
            neighbors = count_neighbors(trajectory, radius=cut).flatten()
            means[j, i] = np.mean(neighbors)

            m = np.max(neighbors)
            if cut > 30:
                bins = np.arange(0, 5 * (m // 5) + 5, 5)
            else:
                bins = np.linspace(0, m, m + 1)
            ax[i].hist(neighbors, histtype='step', label=name, bins=bins, density=True)
    ax[-1].legend(loc='center left', bbox_to_anchor=(1, 0.5))
    for i, cut in enumerate(cuts):
        ax[i].set_title(f'{cut} nm')
        ax[i].set_yticklabels([])
    plt.show()

    plt.plot(cuts, means.T, label=names)
    plt.legend()
    plt.show()


def plot_distance_histograms(conditions, separations=(2, 3, 4, 5, 10, 30, 100, 300)):
    res = 1
    bp_per_bead = 200

    f, ax = plt.subplots(1, len(separations), figsize=(4 * len(separations), 3))
    for name, info in conditions.items():
        positions = load_polymer(info['psf'], info['dcd'], info['selection'], skip=100, scale=info['scale'], pad=100,
                                 burnin=0.0)
        positions = cg_trajectory(positions, res=res)
        for i, separation in enumerate(separations):
            distances = get_distances(positions, separation)
            ax[i].hist(distances.flatten(), histtype='step', density=True, label=name)

    for i, separation in enumerate(separations):
        ax[i].set_title(separation * bp_per_bead * res)
        ax[i].set_yticklabels([])
    ax[-1].legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.show()

######################################################
# Core metrics code.
# `trajectory` is an array of frames, shape (n_frames, n_particles, 3)
# `positions` is an array of particle positions (n_particles, 3)


# Dynamics
def get_single_particle_msds(trajectory, lag=1):
    return np.mean(np.sum((trajectory[lag:] - trajectory[:-lag])**2, axis=-1), axis=0)


def get_single_particle_diffusion_parameters(trajectory, max_lag=10, rc_lag=800):
    lags = np.arange(1, max_lag)
    all_msds = []
    for lag in lags:
        all_msds += [get_single_particle_msds(trajectory, lag=lag)]
    all_msds = np.array(all_msds)

    parameters = []
    for i in range(all_msds.shape[1]):
        popt, pcov = curve_fit(power_law, lags, all_msds[:, i])
        parameters += [popt]
    parameters = np.array(parameters)

    rc = np.sqrt(5 / 6 * get_single_particle_msds(trajectory, lag=rc_lag))
    return np.hstack([parameters, rc.reshape(-1, 1)])


def get_congruence_coefficient(trajectory, remove_com_motion=True, remove_rotation=True):
    if remove_com_motion:
        trajectory = remove_center_of_mass_motion(trajectory, remove_rotation=remove_rotation)

    trajectory = trajectory - np.mean(trajectory, axis=0, keepdims=True)  # Remove center of mass of each particle
    num = np.zeros((trajectory.shape[1], trajectory.shape[1]))
    for t in range(trajectory.shape[0]):
        num += np.matmul(trajectory[t], trajectory[t].T)

    den = np.sum(trajectory**2, axis=(0, -1))
    den = np.sqrt(np.expand_dims(den, -1) * np.expand_dims(den, -2))
    return num / den


def get_average_distance(trajectory):
    n = trajectory.shape[0]
    distances = np.zeros((trajectory.shape[1], trajectory.shape[1]))
    for positions in trajectory:
        distances += distance_matrix(positions, positions)
    return distances / n


def get_all_two_particle_msds(trajectory, lag=1):
    msds = np.zeros((trajectory.shape[1], trajectory.shape[1]))
    n = trajectory.shape[0] - lag
    for t in range(0, trajectory.shape[0] - lag):
        positions1 = trajectory[t]
        positions2 = trajectory[t + lag]
        displacements1 = np.expand_dims(positions1, -2) - np.expand_dims(positions1, -3)
        displacements2 = np.expand_dims(positions2, -2) - np.expand_dims(positions2, -3)
        msds += np.sum((displacements2 - displacements1)**2, axis=-1)
    return msds / n


def get_two_particle_msds(trajectory, sep=10, lag=1):
    msds = np.zeros(trajectory.shape[1] - sep)
    n = trajectory.shape[0] - lag
    for t in range(0, trajectory.shape[0] - lag):
        positions1 = trajectory[t]
        positions2 = trajectory[t + lag]
        displacements1 = positions1[sep:] - positions1[:-sep]
        displacements2 = positions2[sep:] - positions2[:-sep]
        msds += np.sum((displacements2 - displacements1)**2, axis=-1)
    return msds / n


def get_rg(trajectory):
    """
    Compute the radius of gyration of a trajectory.

    Computed as Rg = sqrt(< sum_i (r^{(t)}_i - r^{(cm)})^2 > ).
    """
    assert len(trajectory.shape) == 3  # (n_frames, n_particles, 3)
    center_of_mass = np.mean(trajectory, axis=1, keepdims=True)
    sd = np.sum((trajectory - center_of_mass)**2, axis=2)  # sum over dimensions.
    assert len(sd.shape) == 2  # mean over frames and particles.
    msd = np.mean(sd)
    return np.sqrt(msd)


# Statics.
def get_local_rg(trajectory, window=11, pad=True):
    """
    Compute the average local radius of gyration of a polymer model.

    The local radius of gyration for position i is defined as:
        Rg_i = sqrt(1/w sum_{j=-w//2}^{w//2} (r_{i+j} - r^{(cm)}_i)^2),
            where r^{(cm)}_i = 1/w sum_{j=-w//2}^{w//2} r_{i+j}

    The reported value is the average of Rg_i over all frames.
    """
    assert window % 2 == 1

    rg = np.zeros(trajectory.shape[1])
    s = window // 2 + 1
    e = trajectory.shape[1] - window // 2 - 1
    for i in range(s, e):
        rg[i] = get_rg(trajectory[:, i - window // 2 - 1:i + window // 2 + 2])

    if pad:
        for i in range(s):
            rg[i] = rg[s]
        for i in range(e, len(rg)):
            rg[i] = rg[e - 1]
    return rg


def get_accessibility(trajectory, distance_thresh=30, count_thresh=6) -> np.ndarray:
    """
    Compute the accessibility of each particle in a polymer model.

    In a given frame, a particle is accessible if there are fewer than `count_thresh` particles within `distance_thresh`
    of it. The reported value is the fraction of frames in which a particle is accessible.

    A reasonable choice for distance_thresh is 30 nm, which is approximately the radius of a
    transcription factor or Tn5 transposase. A reasonable choice for count_thresh is
    5-7 which corresponds to the number of 30 nm particles that could surround a particle.
    """
    contacts = np.zeros(trajectory.shape[1])
    for positions in trajectory:
        distances = get_distance_matrix(positions)
        contacts += (distances < distance_thresh).sum(axis=0) < count_thresh
    contacts /= trajectory.shape[0]
    return contacts


def get_accessibility_centers(trajectory, distance_thresh=30, count_thresh=6) -> np.ndarray:
    """
    Compute the accessibility of each particle in a polymer model.

    This implementation considers the accessibility of the centroid of each adjacent pair of particles.

    In a given frame, a particle is accessible if there are fewer than `count_thresh` particles within `distance_thresh`
    of it. The reported value is the fraction of frames in which a particle is accessible.

    A reasonable choice for distance_thresh is 30 nm, which is approximately the radius of a
    transcription factor or Tn5 transposase. A reasonable choice for count_thresh is
    5-7 which corresponds to the number of 30 nm particles that could surround a particle.
    """
    contacts = np.zeros(trajectory.shape[1] - 1)
    for positions in trajectory:
        centers = (positions[:-1] + positions[1:]) / 2
        distances = get_distance_matrix(centers, positions)
        contacts += (distances < distance_thresh).sum(axis=1) < count_thresh
    contacts /= trajectory.shape[0]
    # TODO: This is a hack to make the output the same shape as the other function, but really it should be n_atoms-1
    contacts = (np.concatenate([contacts, [0]]) + np.concatenate([[0], contacts])) / 2
    return contacts


def get_accessibility_surface(trajectory, probe_radius) -> np.ndarray:
    contacts = np.zeros(trajectory.shape[1] - 1)
    for positions in trajectory:
        centers = (positions[1:] + positions[:-1]) / 2
        quarter1 = (3*positions[1:] + positions[:-1]) / 4
        quarter2 = (positions[1:] + 3*positions[:-1]) / 4
        _positions = np.vstack([positions, centers, quarter1, quarter2])

        p, i = solvent_accessible_surface(_positions, 5.5*np.ones(_positions.shape[0]), probe_radius)
        contacts += np.array([np.sum(i == x) for x in range(positions.shape[0], 2*positions.shape[0] - 1)]).astype(float)
    contacts /= trajectory.shape[0]
    contacts = (np.concatenate([contacts, [0]]) + np.concatenate([[0], contacts])) / 2
    return contacts


def get_distances(trajectory, separation):
    distances = []
    for positions in trajectory:
        distances += [np.linalg.norm(positions[:-separation] - positions[separation:], axis=1)]
    return np.array(distances)


def get_angles(trajectory):
    angles = []
    for positions in trajectory:
        a = positions[:-2] - positions[1:-1]
        b = positions[2:] - positions[1:-1]
        angles += [np.arccos(np.sum(a * b, axis=1) / (np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1)))]
    return np.array(angles) * 180 / np.pi



# Concentrations.
def get_local_concentration(trajectory, radius=42):
    contacts = np.zeros(trajectory.shape[1])
    for positions in trajectory:
        contacts += (get_distance_matrix(positions) < radius).sum(axis=0)
    contacts /= trajectory.shape[0]
    v = 4/3 * np.pi * radius**3
    return to_micromolar(contacts, v)


def get_convex_hull_concentration(trajectory, plot=False):
    volumes = []
    for positions in trajectory:
        volumes += [ConvexHull(positions).volume]
    volumes = np.array(volumes)
    if plot:
        median = np.median(volumes)
        plt.axvline(median, c='k', ls='--')
        plt.hist(volumes)
        plt.xlabel('Volume (nm^3)')
        plt.show()

    concentration = to_micromolar(trajectory.shape[1], volumes)
    if plot:
        median = np.median(concentration)
        plt.axvline(median, c='k', ls='--')
        plt.hist(concentration)
        plt.xlabel('Nucleosome Concentration (µM)')
        plt.show()
    return concentration


def voronoi_volumes(positions):
    v = Voronoi(positions)
    volumes = []
    for region in v.regions:
        if -1 not in region and len(region) > 0:
            volumes.append(ConvexHull(v.vertices[region]).volume)
    return np.array(volumes)


def get_voronoi_concentrations_per_frame(trajectory, smp_rate=200, resamples=1):
    concentrations = []
    for positions in trajectory:
        for _ in range(resamples):
            mask = [np.random.random() < 200 / smp_rate for _ in range(positions.shape[0])]
            volumes = voronoi_volumes(positions[mask])
            concentrations += [smp_rate * 10 ** 3 / volumes]
    return concentrations


def get_voronoi_concentrations(trajectory, smp_rate=200, resamples=1):
    concentrations = get_voronoi_concentrations_per_frame(trajectory, smp_rate=smp_rate, resamples=resamples)
    return np.hstack(concentrations).flatten()


def get_overall_voronoi_concentrations(trajectory):
    concentrations = []
    for positions in trajectory:
        volumes = voronoi_volumes(positions)
        volumes = volumes[volumes < np.percentile(volumes, 80)]  # remove outliers
        concentrations += [to_micromolar(len(volumes), sum(volumes))]
    return np.array(concentrations)

############################################################################################


class ProbeVolume:
    def __init__(self, positions, particle_radius, probe_radius, grid_resolution,
                 n_points_per_particle=100, max_neighbors=10, eps=1e-6):
        self.positions = positions
        self.particle_radius = particle_radius
        self.probe_radius = probe_radius
        self.grid_resolution = grid_resolution
        self.n_points_per_particle = n_points_per_particle
        self.max_neighbors = max_neighbors
        self.eps = eps

        self._positions_kdtree = cKDTree(self.positions)
        self._surface_points = None

    def get_volume(self, return_grid_points=False):
        surface_points = self.solvent_accessible_surface()

        radius = self.particle_radius + self.probe_radius

        def get_grid(points):
            low = points.min() - radius
            high = points.max() + radius
            high += self.grid_resolution - ((high - low) % self.grid_resolution)
            n = round((high - low) / self.grid_resolution)
            return np.linspace(low, high, n)

        X = get_grid(self.positions[:, 0])
        Y = get_grid(self.positions[:, 1])
        Z = get_grid(self.positions[:, 2])
        grid_points = np.array(np.meshgrid(X, Y, Z)).reshape(3, -1).T

        # First find the points that are inaccessible to the probe centers,
        # as well as the probe centers that are nearby the interface.
        distances, _ = self._positions_kdtree.query(grid_points, distance_upper_bound=radius)
        inaccessible_to_centers = grid_points[distances > radius]

        # Now remove the points that can be accessed from the probe centers.
        tree = cKDTree(surface_points)
        distances, _ = tree.query(inaccessible_to_centers, distance_upper_bound=self.probe_radius)
        inaccessible = distances > self.probe_radius

        volume = np.sum(inaccessible) * self.grid_resolution ** 3
        if return_grid_points:
            return volume, inaccessible_to_centers[inaccessible], inaccessible_to_centers
        return volume

    def solvent_accessible_surface(self):
        """
        Compute a set of points on the solvent-accessible surface of the particles.

        Returns
        -------
        np.ndarray
            The first ndarray gives the coordinates of each point on the solvent-accessible surface. Shape: (n, 3)
        """
        if self._surface_points is None:
            candidate_surface_points, particle_indices = generate_candidate_surface_points()
            mask = valid_candidate_surface_points(candidate_surface_points)
            self._surface_points = candidate_surface_points[mask]
        return self._surface_points

    def unit_candidate_surface_points(self):
        """
        Generate a set of n points on the unit sphere that are approximately uniformly distributed.

        Parameters
        ----------
        n : int
            The number of points to generate.
        Returns
        -------
        np.ndarray
            The coordinates of the unit candidate surface points. Shape: (n, 3)
        """
        inc = np.pi * (3 - np.sqrt(5))
        offset = 2 / self.n_points_per_particle
        i = np.arange(self.n_points_per_particle)
        y = i * offset - 1 + offset / 2
        r = np.sqrt(1 - y ** 2)
        phi = i * inc
        x = np.cos(phi) * r
        z = np.sin(phi) * r
        return np.stack([x, y, z], axis=1)

    def generate_candidate_surface_points(self, positions, n_points_per_particle, eps=1e-6):
        """
        Generate candidate surface points for each particle.

        Parameters
        ----------
        n_points_per_particle : int
            The number of candidate surface points to generate for each particle.
        eps : float
            A small number to avoid collisions between the candidate surface points the particle it belongs to.

        Returns
        -------
        np.ndarray, np.ndarray
            The first ndarray gives the coordinates of each candidate surface point.
                Shape: (n * n_points_per_particle, 3)
            The second ndarray gives the atom index of the particle that each point belongs to.
                Shape: (n * n_points_per_particle,)
        """
        unit_points = unit_candidate_surface_points(n_points_per_particle)
        points = positions.reshape(-1, 1, 3) + (eps + self.probe_radius + self.particle_radius) * unit_points
        return points.reshape(-1, 3), np.repeat(np.arange(len(positions)), len(unit_points))

    def valid_candidate_surface_points(self, candidate_surface_points):
        """
        Check if candidate surface points are indeed on the solvent-accessible surface.

        Parameters
        ----------
        candidate_surface_points : np.ndarray
            The coordinates of the candidate surface points. Shape: (m, 3)

        Returns
        -------
        np.ndarray
            A boolean mask indicating which candidate surface points are valid. Shape: (m,)
        """
        radius = self.particle_radius + self.probe_radius
        distances, indices = self._positions_kdtree.query(candidate_surface_points, distance_upper_bound=radius)
        return np.all(distances >= radius, axis=1)

    def distance_from_surface(self):
        """
        Compute the distance of each point to the solvent-accessible surface.

        Particles that define the surface give a distance of 0, not the probe radius.

        Returns
        -------
        np.ndarray
            The distance of each point to the solvent-accessible surface. Shape: (n,)
        """
        tree = cKDTree(self.solvent_accessible_surface())
        distances, _ = tree.query(self.positions, k=1)
        return distances - self.probe_radius

    def distances_from_surface(self, trajectory, radii, probe_radius, n_points_per_particle=100, max_neighbors=10,
                               eps=1e-6):
        dists = []
        for positions in trajectory:
            surface_points, particles = solvent_accessible_surface(positions,
                                                                   radii * np.ones(positions.shape[0]),
                                                                   probe_radius,
                                                                   n_points_per_particle=n_points_per_particle,
                                                                   max_neighbors=max_neighbors,
                                                                   eps=eps)
            dists += [distance_from_surface(positions, surface_points, 50)]
        return np.array(dists)


# Surfaces.
def unit_candidate_surface_points(n):
    """
    Generate a set of n points on the unit sphere that are approximately uniformly distributed.

    Parameters
    ----------
    n : int
        The number of points to generate.
    Returns
    -------
    np.ndarray
        The coordinates of the unit candidate surface points. Shape: (n, 3)
    """
    inc = np.pi * (3 - np.sqrt(5))
    offset = 2 / n
    i = np.arange(n)
    y = i * offset - 1 + offset / 2
    r = np.sqrt(1 - y**2)
    phi = i * inc
    x = np.cos(phi) * r
    z = np.sin(phi) * r
    return np.stack([x, y, z], axis=1)


def generate_candidate_surface_points(positions, radii, probe_radius, n_points_per_particle, eps=1e-6):
    """
    Generate candidate surface points for each particle.

    Parameters
    ----------
    positions : np.ndarray
        The coordinates of the particles. Shape: (n, 3)
    radii : np.ndarray
        The radii of the particles. Shape: (n,)
    probe_radius : float
        The radius of the probe.
    n_points_per_particle : int
        The number of candidate surface points to generate for each particle.
    eps : float
        A small number to avoid collisions between the candidate surface points the particle it belongs to.

    Returns
    -------
    np.ndarray, np.ndarray
        The first ndarray gives the coordinates of each candidate surface point.
            Shape: (n * n_points_per_particle, 3)
        The second ndarray gives the atom index of the particle that each point belongs to.
            Shape: (n * n_points_per_particle,)
    """
    unit_points = unit_candidate_surface_points(n_points_per_particle)
    points = positions.reshape(-1, 1, 3) + (eps + probe_radius + radii.reshape(-1, 1, 1)) * unit_points
    return points.reshape(-1, 3), np.repeat(np.arange(len(positions)), len(unit_points))


def valid_candidate_surface_points_reference(positions, radii, probe_radius, candidate_surface_points):
    """
    Check if candidate surface points are indeed on the solvent-accessible surface.

    This implementation is a reference implementation that is not optimized for performance. Only use it to verify
    the correctness of the optimized implementation.
    """
    distances = np.linalg.norm(candidate_surface_points.reshape(-1, 1, 3) - positions, axis=-1)
    return np.all(distances >= radii + probe_radius, axis=1)


def valid_candidate_surface_points(positions, radii, probe_radius, candidate_surface_points, max_neighbors=10):
    """
    Check if candidate surface points are indeed on the solvent-accessible surface.

    Parameters
    ----------
    positions : np.ndarray
        The coordinates of the particles. Shape: (n, 3)
    radii : np.ndarray
        The radii of the particles. Shape: (n,)
    probe_radius : float
        The radius of the probe.
    candidate_surface_points : np.ndarray
        The coordinates of the candidate surface points. Shape: (m, 3)
    max_neighbors : int
        The maximum number of neighbors to consider when checking for collisions.

    Returns
    -------
    np.ndarray
        A boolean mask indicating which candidate surface points are valid. Shape: (m,)
    """
    tree = cKDTree(positions)
    distances, indices = tree.query(candidate_surface_points, k=max_neighbors)
    return np.all(distances >= radii[indices] + probe_radius, axis=1)


def solvent_accessible_surface(positions, radii, probe_radius, n_points_per_particle=100, max_neighbors=10, eps=1e-6):
    """
    Compute a set of points on the solvent-accessible surface of the particles.

    Returns
    -------
    np.ndarray, np.ndarray
        The first ndarray gives the coordinates of each point on the solvent-accessible surface. Shape: (n, 3)
        The second ndarray gives the atom index of the particle that each point belongs to. Shape: (n,)
    """
    candidate_surface_points, particle_indices = generate_candidate_surface_points(positions, radii, probe_radius,
                                                                            n_points_per_particle=n_points_per_particle,
                                                                            eps=eps)
    mask = valid_candidate_surface_points(positions, radii, probe_radius, candidate_surface_points,
                                          max_neighbors=max_neighbors)
    return candidate_surface_points[mask], particle_indices[mask]


def distance_from_surface(positions, surface_points, probe_radius):
    """
    Compute the distance of each point to the solvent-accessible surface.

    Particles that define the surface give a distance of 0, not the probe radius.

    Parameters
    ----------
    positions : np.ndarray
        The coordinates of the points. Shape: (n, 3)
    surface_points : np.ndarray
        The coordinates of the solvent-accessible surface points. Shape: (m, 3)
    probe_radius : float
        The radius of the probe. This is subtracted from the distances.

    Returns
    -------
    np.ndarray
        The distance of each point to the solvent-accessible surface. Shape: (n,)
    """
    tree = cKDTree(surface_points)
    distances, _ = tree.query(positions, k=1)
    return distances - probe_radius


def distances_from_surface(trajectory, radii, probe_radius, n_points_per_particle=100, max_neighbors=10, eps=1e-6):
    dists = []
    for positions in trajectory:
        surface_points, particles = solvent_accessible_surface(positions,
                                                               radii * np.ones(positions.shape[0]),
                                                               probe_radius,
                                                               n_points_per_particle=n_points_per_particle,
                                                               max_neighbors=max_neighbors,
                                                               eps=eps)
        dists += [distance_from_surface(positions, surface_points, probe_radius)]
    return np.array(dists)

############################################################################################


def read_4DNFI9J3L3G9(fname):
    # This one is in a different format for whatever reason.
    df = pd.read_csv(fname, comment='#', skiprows=10, low_memory=False)
    df.rename(columns={'x': 'X', 'y': 'Y', 'z': 'Z', 'chr_start': 'Chrom_Start', 'trace_ID': 'Trace_ID'}, inplace=True)
    df = df.sort_values(['Trace_ID', 'Chrom_Start'])
    return df


def read_4dn(fname):
    if '4DNFI9J3L3G9' in fname:
        return read_4DNFI9J3L3G9(fname)
    with open(fname) as f:
        for line in f:
            if 'columns=' in line:
                columns = line.split('(')[1].split(')')[0].split(',')
                columns = [x.strip() for x in columns]
                break
        else:
            columns = None

    df = pd.read_csv(fname, comment='#', names=columns, header=None, low_memory=False)
    df = df.loc[~np.isnan(df['Trace_ID'])]
    return df


def get_4dn_distances(df):
    p, g = [], []
    for i, group in df.groupby('Trace_ID'):
        physical = group[['X', 'Y', 'Z']].to_numpy()
        genomic = group[['Chrom_Start']].to_numpy()
        physical_distances = np.linalg.norm(physical.reshape(-1, 1, 3) - physical.reshape(1, -1, 3), axis=2)
        genomic_distances = np.abs(genomic.reshape(-1, 1) - genomic.reshape(1, -1))
        p += [full_to_triu(physical_distances, k=1)]
        g += [full_to_triu(genomic_distances, k=1)]
    return np.hstack(p), np.hstack(g)
