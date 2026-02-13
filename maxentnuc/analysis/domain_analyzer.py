from sklearn.cluster import DBSCAN
from scipy.spatial import ConvexHull, cKDTree
from matplotlib import colormaps
from matplotlib.colors import ListedColormap, Normalize
from matplotlib.collections import LineCollection
from matplotlib.gridspec import GridSpec
import matplotlib.pyplot as plt
from .analysis import to_micromolar
from .analysis import get_distance_matrix
import numpy as np
import plotly.graph_objects as go
from maxentnuc.analysis.mei_analyzer import MEIAnalyzer
from neighbor_balance.plotting import ContactMap, format_ticks, parse_region
import click
from sklearn.decomposition import PCA


# TODO: this is the beginning of a more efficient version of `get_volume`.
#  If I want to use it, I need to write tests and finish it...
# class ProbeVolume:
#     def __init__(self, positions, particle_radius, probe_radius, grid_resolution,
#                  n_points_per_particle=100, max_neighbors=10, eps=1e-6):
#         self.positions = positions
#         self.particle_radius = particle_radius
#         self.probe_radius = probe_radius
#         self.grid_resolution = grid_resolution
#         self.n_points_per_particle = n_points_per_particle
#         self.max_neighbors = max_neighbors
#         self.eps = eps
#
#         self._positions_kdtree = cKDTree(self.positions)
#         self._surface_points = None
#
#     def double_grid(self, grid_points, old_resolution):
#         # Break each grid volume into 8 cubic sub volumes.
#         added = []
#         for i in [-1, 1]:
#             for j in [-1, 1]:
#                 for k in [-1, 1]:
#                     added += [[i, j, k]]
#         added = np.array(added) * old_resolution / 4
#
#         new_grid = grid_points[:, None, :] + added[None, :, :]
#
#         return new_grid.reshape(-1, 3)
#
#     def get_volume(self, return_grid_points=False):
#         radius = self.particle_radius + self.probe_radius
#
#         iterations = 0
#
#         def get_grid(points):
#             res = self.grid_resolution * 2 ** iterations
#             low = points.min() - radius
#             high = points.max() + radius
#             high += self.grid_resolution - ((high - low) % res)
#             n = round((high - low) / res)
#             return np.linspace(low, high, n)
#
#         X = get_grid(self.positions[:, 0])
#         Y = get_grid(self.positions[:, 1])
#         Z = get_grid(self.positions[:, 2])
#         grid_points = np.array(np.meshgrid(X, Y, Z)).reshape(3, -1).T
#         print(grid_points.shape)
#         for i in range(iterations, 0, -1):
#             r = radius + self.grid_resolution * 2 ** i
#             distances, _ = self._positions_kdtree.query(grid_points, distance_upper_bound=r)
#             grid_points = grid_points[distances < r]
#             print('pruned', grid_points.shape)
#             grid_points = self.double_grid(grid_points, self.grid_resolution * 2 ** i)
#             print('doubled', grid_points.shape)
#
#         # First find the points that are inaccessible to the probe centers,
#         # as well as the probe centers that are nearby the interface.
#         distances, _ = self._positions_kdtree.query(grid_points, distance_upper_bound=radius)
#         inaccessible_to_centers = grid_points[distances < radius]
#
#         # Now remove the points that can be accessed from the probe centers.
#         surface_points = self.solvent_accessible_surface()
#         tree = cKDTree(surface_points)
#         distances, _ = tree.query(inaccessible_to_centers, distance_upper_bound=self.probe_radius)
#         inaccessible = distances > self.probe_radius
#
#         volume = np.sum(inaccessible) * self.grid_resolution ** 3
#         if return_grid_points:
#             return volume, inaccessible_to_centers[inaccessible], inaccessible_to_centers
#         return volume
#
#     def solvent_accessible_surface(self):
#         """
#         Compute a set of points on the solvent-accessible surface of the particles.
#
#         Returns
#         -------
#         np.ndarray
#             The first ndarray gives the coordinates of each point on the solvent-accessible surface. Shape: (n, 3)
#         """
#         if self._surface_points is None:
#             candidate_surface_points, particle_indices = self.generate_candidate_surface_points()
#             mask = self.valid_candidate_surface_points(candidate_surface_points)
#             self._surface_points = candidate_surface_points[mask]
#         return self._surface_points
#
#     def unit_candidate_surface_points(self):
#         """
#         Generate a set of n points on the unit sphere that are approximately uniformly distributed.
#
#         Parameters
#         ----------
#         n : int
#             The number of points to generate.
#         Returns
#         -------
#         np.ndarray
#             The coordinates of the unit candidate surface points. Shape: (n, 3)
#         """
#         inc = np.pi * (3 - np.sqrt(5))
#         offset = 2 / self.n_points_per_particle
#         i = np.arange(self.n_points_per_particle)
#         y = i * offset - 1 + offset / 2
#         r = np.sqrt(1 - y ** 2)
#         phi = i * inc
#         x = np.cos(phi) * r
#         z = np.sin(phi) * r
#         return np.stack([x, y, z], axis=1)
#
#     def generate_candidate_surface_points(self):
#         """
#         Generate candidate surface points for each particle.
#
#         Parameters
#         ----------
#         n_points_per_particle : int
#             The number of candidate surface points to generate for each particle.
#         eps : float
#             A small number to avoid collisions between the candidate surface points the particle it belongs to.
#
#         Returns
#         -------
#         np.ndarray, np.ndarray
#             The first ndarray gives the coordinates of each candidate surface point.
#                 Shape: (n * n_points_per_particle, 3)
#             The second ndarray gives the atom index of the particle that each point belongs to.
#                 Shape: (n * n_points_per_particle,)
#         """
#         unit_points = self.unit_candidate_surface_points()
#         points = positions.reshape(-1, 1, 3) + (self.eps + self.probe_radius + self.particle_radius) * unit_points
#         return points.reshape(-1, 3), np.repeat(np.arange(len(positions)), len(unit_points))
#
#     def valid_candidate_surface_points(self, candidate_surface_points):
#         """
#         Check if candidate surface points are indeed on the solvent-accessible surface.
#
#         Parameters
#         ----------
#         candidate_surface_points : np.ndarray
#             The coordinates of the candidate surface points. Shape: (m, 3)
#
#         Returns
#         -------
#         np.ndarray
#             A boolean mask indicating which candidate surface points are valid. Shape: (m,)
#         """
#         radius = self.particle_radius + self.probe_radius
#         distances, _ = self._positions_kdtree.query(candidate_surface_points, distance_upper_bound=radius)
#         return distances >= radius
#
#     def distance_from_surface(self):
#         """
#         Compute the distance of each point to the solvent-accessible surface.
#
#         Particles that define the surface give a distance of 0, not the probe radius.
#
#         Returns
#         -------
#         np.ndarray
#             The distance of each point to the solvent-accessible surface. Shape: (n,)
#         """
#         tree = cKDTree(self.solvent_accessible_surface())
#         distances, _ = tree.query(self.positions, k=1)
#         return distances - self.probe_radius
#
#     def distances_from_surface(self, trajectory, radii, probe_radius, n_points_per_particle=100, max_neighbors=10,
#                                eps=1e-6):
#         dists = []
#         for positions in trajectory:
#             surface_points, particles = solvent_accessible_surface(positions,
#                                                                    radii * np.ones(positions.shape[0]),
#                                                                    probe_radius,
#                                                                    n_points_per_particle=n_points_per_particle,
#                                                                    max_neighbors=max_neighbors,
#                                                                    eps=eps)
#             dists += [distance_from_surface(positions, surface_points, 50)]
#         return np.array(dists)


def get_volume(positions, particle_radius, probe_radius, grid_resolution, return_grid_points=False):
    """
    Calculate the volume of the space that is inaccessible to a probe of a given radius.

    Parameters
    ----------
    positions : np.ndarray
        A 2D array of particle positions.
    particle_radius : float
        The radius of the particles in nm.
    probe_radius : float
        The radius of the probe in nm.
    grid_resolution : float
        The resolution of the grid in nm.
    return_grid_points : bool
        If True, return the grid points that are inaccessible to the probes.

    Returns
    -------
    float or tuple
        The volume of the space that is inaccessible to the probe in nm^3. If return_grid_points is True,
        a tuple is returned with the volume and the grid points that are inaccessible.
    """
    radius = particle_radius + probe_radius

    def get_grid(points):
        low = points.min() - radius
        high = points.max() + radius
        high += grid_resolution - ((high - low) % grid_resolution)
        n = round((high - low) / grid_resolution)
        return np.linspace(low, high, n)

    X = get_grid(positions[:, 0])
    Y = get_grid(positions[:, 1])
    Z = get_grid(positions[:, 2])
    grid_points = np.array(np.meshgrid(X, Y, Z)).reshape(3, -1).T

    # First find the points that are inaccessible to the probe centers,
    # as well as the probe centers that are nearby the interface.
    tree = cKDTree(positions)
    distances, _ = tree.query(grid_points, distance_upper_bound=radius+grid_resolution)
    nearby_probe_centers = grid_points[(distances > radius) & (distances < radius + grid_resolution)]
    inaccessible_to_centers = grid_points[distances < radius]

    # Now remove the points that can be accessed from the probe centers.
    tree = cKDTree(nearby_probe_centers)
    distances, _ = tree.query(inaccessible_to_centers, distance_upper_bound=probe_radius)
    inaccessible = distances > probe_radius

    volume = np.sum(inaccessible) * grid_resolution ** 3
    if return_grid_points:
        return volume, inaccessible_to_centers[inaccessible], inaccessible_to_centers
    return volume


class Domains:
    """
    A class to represent and analyze the results of clustering.

    Attributes
    ----------
    positions : np.ndarray
        A 2D array of particle positions.
    labels : np.ndarray
        Cluster labels assigned by DBSCAN.
    volume_method : str
        Method used to calculate the volume of clusters.
    volume_kwargs : dict
        Additional arguments for volume calculation.
    """
    def __init__(self, positions, labels, compact_labels=True):
        self.positions = positions
        self.labels = self.compact_labels(labels) if compact_labels else labels

        self.volume_method = 'probe'
        self.volume_kwargs = {'grid_resolution': 10, 'probe_radius': 20, 'particle_radius': 5.5}
        self._volumes = None  # Cache for volumes to avoid recomputation.

    @classmethod
    def compact_labels(cls, labels):
        new_labels = np.zeros(labels.shape, dtype=int)
        new_labels[labels == -1] = -1
        for new_cluster, old_cluster in enumerate(np.unique(labels[labels != -1])):
            new_labels[labels == old_cluster] = new_cluster
        return new_labels

    def set_volume_mode(self, method, kwargs=None):
        self.volume_method = method
        if kwargs is None:
            self.volume_kwargs = {}
        else:
            self.volume_kwargs = kwargs
        self._volumes = None

    @property
    def n_clusters(self):
        """Number of clusters detected (excluding noise)."""
        return max(self.labels) + 1

    @property
    def n_noise(self):
        """Number of noise points (labeled as -1)."""
        return np.sum(self.labels == -1)

    def counts(self):
        """Calculate the number of points per cluster."""
        counts = []
        for i in range(self.n_clusters):
            counts += [np.sum(self.labels == i)]
        return counts

    def volumes(self, min_points=4):
        """Calculate the volumes of clusters."""
        if self._volumes is None:
            volumes = []
            for i in range(self.n_clusters):
                points = self.positions[self.labels == i]
                if len(points) < min_points:
                    volumes += [np.nan]
                else:
                    if self.volume_method == 'convex_hull':
                        hull = ConvexHull(points)
                        volumes += [hull.volume]
                    elif self.volume_method == 'probe':
                        volumes += [get_volume(points, **self.volume_kwargs)]
                    else:
                        raise ValueError(f"Unknown volume method: {self.volume_method}")
            self._volumes = np.array(volumes)
        return self._volumes.copy()

    def diameters(self, mode='volume'):
        """Calculate the diameters of clusters based on their volume."""
        if mode == 'volume':
            return [2 * (volume / (4 / 3 * np.pi)) ** (1 / 3) for volume in self.volumes()]
        elif mode == 'projection':
            diameters = []
            for i in range(self.n_clusters):
                points = self.positions[self.labels == i, :2]
                cov_matrix = np.cov(points.T)
                eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)

                # Step 5: Compute the lengths of the axes
                major_length = np.sqrt(np.max(eigenvalues))
                minor_length = np.sqrt(np.min(eigenvalues))
                diameters += [major_length]
            return diameters

    def _densities(self):
        """Calculate the density of clusters."""
        densities = []
        for n, d in zip(self.counts(), self.diameters()):
            concentration = n / (4 / 3 * np.pi * (d / 2) ** 3)  # count / nm^3
            concentration /= 6.022 * 10 ** 23  # moles / nm^3
            concentration *= 1.0 * 10 ** 24  # N / dm^3
            concentration *= 10 ** 6  # µM
            densities += [concentration]
        return densities

    def densities(self, min_diameter=0, method='volume', radius=40):
        if method == 'volume':
            densities = self._densities()
        elif method == 'max_concentration':
            conc = self.get_concentrations(radius=radius)
            densities = [np.max(conc[self.labels == i]) for i in range(self.n_clusters)]
        elif method == 'average_concentration':
            conc = self.get_concentrations(radius=radius)
            densities = [np.mean(conc[self.labels == i]) for i in range(self.n_clusters)]
        else:
            raise ValueError(f"Unknown density method: {method}")
        diameters = self.diameters()
        return [density for density, diameter in zip(densities, diameters) if diameter > min_diameter]

    def info(self):
        """Generate a summary of the clustering results."""
        diameters = self.diameters()
        return (f'Number of clusters: {self.n_clusters}\n'
                f'Number of noise points: {self.n_noise}\n'
                f'cluster sizes = {np.min(diameters), np.mean(diameters)}, {np.max(diameters)}')

    def colors(self, noise_color='black'):
        n_colors = 9
        original = colormaps['Set1']
        colors = original(np.linspace(0, 1, n_colors))
        if noise_color == 'black':
            noise = np.array([[0, 0, 0, 1]])
        elif noise_color == 'white':
            noise = np.array([[1, 1, 1, 1]])
        else:
            raise ValueError(f"Unknown noise color: {noise_color}")
        colors = np.vstack(([noise] if self.n_noise > 0 else [])
                           + [colors] * (self.n_clusters // n_colors)
                           + [colors[:self.n_clusters % n_colors]])
        return ListedColormap(colors)

    def plot_all_2D(self, **kwargs):
        f, ax = plt.subplots(1, 3, figsize=(13, 3), sharey=True)

        ax[0].set_title('Chain Index')
        self.plot_chain_2D(ax=ax[0], **kwargs)

        ax[1].set_title('Concentration (µM)')
        self.plot_concentration_2D(ax=ax[1], **kwargs)

        ax[2].set_title('Cluster Index')
        self.plot_2D(ax=ax[2], **kwargs)

    def _plot_2D(self, cmap, colors, pca=True, ax=None, s=5, alpha=0.5, colorbar=True, **kwargs):
        positions = self.positions - self.positions.mean(axis=0)

        if pca:
            if ax is None:
                f, ax = plt.subplots()
            x, y, z = PCA(3).fit_transform(positions).T

            # Center based on extent.
            x -= (np.max(x) + np.min(x)) / 2
            y -= (np.max(y) + np.min(y)) / 2
            z -= (np.max(z) + np.min(z)) / 2
            order = np.argsort(z)
            x = x[order]
            y = y[order]
            z = z[order]
            colors = colors[order]
            sc = ax.scatter(x, y, c=colors, cmap=cmap, s=s, alpha=alpha, rasterized=True, **kwargs)
            if colorbar:
                plt.colorbar(sc)
            ax.set_xlabel('PC1')
            ax.set_ylabel('PC2')
            ax = [ax]
        else:
            x, y, z = positions.T
            f, ax = plt.subplots(1, 3, figsize=(13, 4))
            ax[0].scatter(x, y, zorder=z,c=colors, cmap=cmap, s=s, **kwargs)
            ax[1].scatter(z, y, zorder=x, c=colors, cmap=cmap, s=s, **kwargs)
            sc = ax[2].scatter(x, z, zorder=y, c=colors, cmap=cmap, s=s, **kwargs)
            ax[0].set_ylabel('y')
            ax[0].set_xlabel('x')
            ax[1].set_ylabel('y')
            ax[1].set_xlabel('z')
            ax[2].set_ylabel('z')
            ax[2].set_xlabel('x')
            cbar_ax = f.add_axes([0.92, 0.15, 0.02, 0.7])  # [left, bottom, width, height]
            if colorbar:
                plt.colorbar(sc, cax=cbar_ax)

        # Make all the axes have the same limits and aspect ratio.
        low = min(x.min(), z.min(), y.min()) - 10
        high = max(x.max(), z.max(), y.max()) + 10
        for a in ax:
            a.set_xlim(low, high)
            a.set_ylim(low, high)
            a.set_aspect('equal')
        return ax

    def _plot_3D(self, colors, cmap=None, size=600, s=3):
        fig = go.Figure()
        fig.add_trace(go.Scatter3d(
            x=self.positions[:, 0],
            y=self.positions[:, 1],
            z=self.positions[:, 2],
            mode='lines',
            line=dict(
                color='black',
                width=2
            )
        ))
        fig.add_trace(go.Scatter3d(
            x=self.positions[:, 0],
            y=self.positions[:, 1],
            z=self.positions[:, 2],
            mode='markers',
            marker=dict(
                size=s,
                color=colors,
                colorscale=cmap,
                showscale=cmap is not None,
            )
        ))
        fig.update_layout(
            width=size,
            height=size,
        )
        fig.show()

    def plot_2D(self, **kwargs):
        self._plot_2D(cmap=self.colors(), colors=self.labels, **kwargs)

    def plot_3D(self, size=600):
        colors = [f'rgb{tuple(255*(x[:3]))}' for x in self.colors()(Normalize()(self.labels))]
        self._plot_3D(colors=colors, size=size)

    def get_concentrations(self, radius=40):
        dists = get_distance_matrix(self.positions)
        n = np.sum(dists < radius, axis=1)
        v = 4 / 3 * np.pi * radius ** 3
        return to_micromolar(n, v)

    def plot_concentration_2D(self, vmax=500, vmin=0, **kwargs):
        conc = self.get_concentrations()
        self._plot_2D(cmap='viridis_r', colors=conc, vmin=vmin, vmax=vmax, **kwargs)

    def plot_concentration_3D(self, size=600):
        conc = self.get_concentrations()
        self._plot_3D(colors=conc, cmap='viridis', size=size)

    def plot_chain_2D(self, **kwargs):
        self._plot_2D(cmap='turbo', colors=np.arange(self.positions.shape[0]), **kwargs)

    def plot_chain_3D(self, size=600):
        self._plot_3D(colors=np.linspace(0, 1, self.positions.shape[0]), cmap='turbo', size=size)

    def plot_distances(self):
        scale = 0.5
        margin = 0.5
        sep = 0.2
        base = 0.4
        h = 22 * base + 2 * margin + sep
        w = 21 * base + 3 * margin + sep
        left_right = margin / w
        top_bottom = margin / h
        f = plt.figure(figsize=(scale * w, scale * h), dpi=200)
        gs = GridSpec(2, 2, height_ratios=[2, 20], width_ratios=[20, 1],
                      left=2 * left_right, right=1 - left_right, wspace=sep / w,
                      top=1 - top_bottom, bottom=top_bottom, hspace=sep / h)

        image_ax = f.add_subplot(gs[1, 0])
        cbar_ax = f.add_subplot(gs[1, 1].subgridspec(3, 1, height_ratios=[0.25, 1, 0.25])[1])
        plot_ax = f.add_subplot(gs[0, 0], sharex=image_ax)

        res = 200
        start, end = 0, res * self.positions.shape[0]
        region = f'chr1:{start}-{end}'
        im = plot_pairwise(get_distance_matrix(self.positions), region, ax=image_ax, vmin=0, vmax=200, colorbar=False,
                           cmap='viridis')
        plt.colorbar(im, cax=cbar_ax, extend='max')

        x = np.arange(start, end, res)
        y = np.zeros(self.positions.shape[0])
        z = self.labels

        points = np.array([x, y]).T.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)
        lc = LineCollection(segments, cmap=self.colors())
        lc.set_array(z)
        lc.set_linewidth(10)
        plot_ax.add_collection(lc)
        plot_ax.set_ylim(-.1, .1)
        plot_ax.axis('off')
        plt.show()


class DomainAnalyzer:
    def __init__(self, radius, concentration, min_loop_size=2, add_boundary_linkers=True):
        """
        Parameters
        ----------
        radius : float
            The radius of the cluster in nm.
        concentration : float
            The concentration of the particles in µM.
        """
        self.radius = radius
        self.concentration = concentration
        self.min_loop_size = min_loop_size
        self.add_boundary_linkers = add_boundary_linkers

    @property
    def min_samples(self):
        """
        Calculate the minimum number of samples for DBSCAN given the radius and concentration.
        """
        volume = 4 / 3 * np.pi * self.radius ** 3  # nm^3

        n = self.concentration * 1e-6  # M
        n /= 1.0 * 10 ** 24  # moles / nm^3
        n *= 6.022 * 10 ** 23  # count / nm^3
        n *= volume  # count
        n = int(round(n, 0))

        # Just check that the math above is correct...
        concentration = n / volume  # count / nm^3
        concentration /= 6.022 * 10 ** 23  # moles / nm^3
        concentration *= 1.0 * 10 ** 24  # N / dm^3
        concentration *= 10 ** 6  # µM
        assert abs(concentration - self.concentration) < 10
        return n

    def analyze_frame(self, positions):
        dbscan = DBSCAN(eps=self.radius, min_samples=self.min_samples)
        dbscan.fit(positions)
        labels = dbscan.labels_
        if self.add_boundary_linkers:
            labels = self.add_zero_length_linkers(labels)
        if self.min_loop_size > 1:
            labels = self.remove_short_linkers(labels)
        return Domains(positions, labels)

    def analyze_trajectory(self, trajectory):
        return [self.analyze_frame(positions) for positions in trajectory]
    

    def remove_short_linkers(self, labels):
        """
        Remove small linkers between the same cluster.
        """
        loops = get_loops(labels)
        updated_labels = labels.copy()
        for start, end in loops:
            if (end - start) >= self.min_loop_size:
                continue
            if start == 0:
                updated_labels[start:end] = labels[end]
            elif end == len(labels):
                updated_labels[start:end] = labels[start - 1]
            elif labels[start - 1] == labels[end]:
                updated_labels[start:end] = labels[start - 1]
        return updated_labels

    def add_zero_length_linkers(self, labels):
        """
        Add linkers between clusters that are not separated by any noise points (-1).
        """
        NOISE = -1
        cluster_change = (labels[1:] != labels[:-1])
        not_noise = (labels[1:] != NOISE) & (labels[:-1] != NOISE)
        mask = cluster_change & not_noise
        indices = np.arange(len(labels) - 1)[mask]
        updated_labels = labels.copy()
        updated_labels[indices] = NOISE
        updated_labels[indices + 1] = NOISE
        return updated_labels


def get_loops(labels):
    loops = []
    loop_start = None
    for i in range(len(labels)):
        if labels[i] == -1:
            if loop_start is None:
                loop_start = i
        elif loop_start is not None:
            loops += [(loop_start, i)]
            loop_start = None
    if loop_start is not None:
        loops += [(loop_start, len(labels))]
    return loops


def get_boundary_loops(labels):
    loops = get_loops(labels)

    boundary_loops = []
    for start, end in loops:
        if start == 0:
            continue
        if end == len(labels) - 1:
            continue

        if labels[start - 1] != labels[end]:
            boundary_loops += [(start, end)]

    mask = labels[1:] != labels[:-1]
    mask &= labels[1:] != -1
    mask &= labels[:-1] != -1
    no_loops = np.arange(len(mask))[mask]
    for s in no_loops:
        boundary_loops += [(s, s+1)]
    return boundary_loops


def plot_domain_size_trace(domains, region, fname):
    cutoffs = [200, 500, float('inf')]

    counts = np.zeros((len(domains[0].positions), 1 + len(cutoffs)))
    for domain in domains:
        domain.set_volume_mode('convex_hull')
        domain_diameters = domain.diameters()
        diameters_trace = np.zeros(len(domain.positions))
        for cluster in range(domain.n_clusters):
            mask = domain.labels == cluster
            diameters_trace[mask] = domain_diameters[cluster]

        counts[:, 0] += domain.labels == -1
        for i, cutoff in enumerate(cutoffs):
            counts[:, i + 1] += diameters_trace < cutoff
    counts /= len(domains)
    f, ax = plt.subplots(figsize=(7, 2))
    chrom, start, end = parse_region(region)
    x = np.arange(start, end, 200)
    plt.fill_between(x, np.zeros(x.shape), counts[:, 0], alpha=0.75)
    for i in range(counts.shape[1] - 1):
        plt.fill_between(x, counts[:, i], counts[:, i + 1], alpha=0.75)
    format_ticks(ax, y=False)
    plt.savefig(fname)
    plt.close()


def plot_same_domain(domains, region, fname):
    counts = np.zeros((len(domains[0].positions), len(domains[0].positions)))
    for domain in domains:
        same = domain.labels[:, None] == domain.labels[None, :]
        noise = (domain.labels[:, None] == -1) & (domain.labels[None, :] == -1)
        counts += same & ~noise
    counts /= len(domains)
    plot_pairwise(counts, region, vmin=0, vmax=1, cmap='fall')
    plt.savefig(fname)
    plt.close()


def get_sizes(domains):
    sizes = np.zeros((len(domains), (end - start) // 200))
    for t, d in enumerate(domains):
        for i in range(d.n_clusters):
            mask = d.labels == i
            sizes[t, mask] = np.sum(mask)
    return sizes


@click.command()
@click.option('--concentration', default=200)
@click.option('--radius', default=40)
@click.option('--config', default='config.yaml')
@click.option('--scale', default=0.1)
@click.option('--skip', default=1)
@click.option('--iteration', default=-1)
def main(concentration, radius, config, scale, skip, iteration):
    mei = MEIAnalyzer(config, scale=scale)
    region = mei.mei.config['region']

    if iteration == -1:
        iteration = mei.get_iterations()[-1]

    analyzer = DomainAnalyzer(radius, concentration)
    trajectory = mei.get_positions(iteration, skip=skip)
    trajectory = trajectory.reshape(-1, *trajectory.shape[2:])
    domains = analyzer.analyze_trajectory(trajectory)
    plot_domain_size_trace(domains, region, 'domain_trace.pdf')
    plot_same_domain(domains, region, 'domain_map.pdf')


if __name__ == '__main__':
    main()
