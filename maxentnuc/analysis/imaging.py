from scipy.ndimage import gaussian_filter
from skimage import exposure
from skimage.feature import peak_local_max
from scipy.optimize import curve_fit
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec
from scipy.signal import convolve2d
from scipy.ndimage import label


def power_law(x, a, b):
    return a*x**b


def fit_power_law(x, y):
    opt, _ = curve_fit(power_law, x, y)
    return opt

def plot_density(h, ax, resolution, colorbar=False, **kwargs):
    width = resolution * h.shape[0]
    im = ax.imshow(h.T, extent=[0, width, 0, width],
                    origin='lower', cmap='viridis', **kwargs)
    if colorbar:
        plt.colorbar(im)


class ChromSTEMSimulator:
    def __init__(self, width=1000, resolution=2, slice_z=0, slice_thickness=100, nucleosome_radius=5):
        self.width = width
        self.resolution = resolution
        self.slice_z = slice_z
        self.slice_thickness = slice_thickness
        self.nucleosome_radius = nucleosome_radius

        assert width % resolution == 0, "Width must be divisible by resolution."

    def slice_weights(self, positions):
        weights = (positions[:, 2] > self.slice_z - self.slice_thickness / 2) & (positions[:, 2] < self.slice_z + self.slice_thickness / 2)
        return weights.astype(float)

    def simulate(self, positions):
        weights = self.slice_weights(positions)
        h = np.histogram2d(positions[:, 0], positions[:, 1],
                           bins=self.width//self.resolution,
                           weights=weights,
                           range=[[-self.width/2, self.width/2], [-self.width/2, self.width/2]])[0]

        # Smooth with nucleosome size. This could later be replaced with a more accurate PSF.
        h = gaussian_filter(h, sigma=self.nucleosome_radius/self.resolution)

         # Normalize so that the center peak is 1 nucleosome
        h *= np.sqrt(2 * np.pi) * (self.nucleosome_radius / self.resolution)
        return h

    def plot_positions(self, positions, size_3d=5, size_2d=8):
        """
        Left: 3D nucleosome positions with slice.
        Right: 2D projection of nucleosomes inside the slice.

        positions: (N, 3) array
        """

        # Slice mask
        z0 = self.slice_z
        dz = self.slice_thickness / 2.0
        mask = (positions[:, 2] > z0 - dz) & (positions[:, 2] < z0 + dz)
        pos_slice = positions[mask]

        fig = plt.figure(figsize=(9, 4))
        gs = gridspec.GridSpec(1, 2, width_ratios=[2, 1])

        # ---- Left: 3D view ----
        ax3d = fig.add_subplot(gs[0], projection="3d")

        ax3d.scatter(
            positions[:, 0],
            positions[:, 1],
            positions[:, 2],
            s=size_3d,
            alpha=0.25,
            c = np.arange(len(positions)),
            cmap='turbo'
        )

        # Slice planes
        xlim = [positions[:, 0].min(), positions[:, 0].max()]
        ylim = [positions[:, 1].min(), positions[:, 1].max()]

        xx, yy = np.meshgrid(xlim, ylim)
        for z in [z0 - dz, z0 + dz]:
            zz = np.full_like(xx, z)
            ax3d.plot_surface(xx, yy, zz, alpha=0.15, color="red")

        ax3d.set_xlabel("X (nm)")
        ax3d.set_ylabel("Y (nm)")
        ax3d.set_zlabel("Z (nm)")
        ax3d.set_title("3D nucleosome positions with slice")

        # Equal aspect ratio for 3D
        max_range = (positions.max(axis=0) - positions.min(axis=0)).max()
        mid = positions.mean(axis=0)
        ax3d.set_xlim(mid[0] - max_range / 2, mid[0] + max_range / 2)
        ax3d.set_ylim(mid[1] - max_range / 2, mid[1] + max_range / 2)
        ax3d.set_zlim(mid[2] - max_range / 2, mid[2] + max_range / 2)
        ax3d.set_box_aspect([1,1,1])  # Equal aspect ratio
        ax3d.view_init(elev=10., azim=45)

        # ---- Right: 2D projection ----
        ax2d = fig.add_subplot(gs[1])

        ax2d.scatter(
            pos_slice[:, 0],
            pos_slice[:, 1],
            s=size_2d,
            alpha=0.8,
            c=np.arange(len(positions))[mask],
            cmap='turbo'
        )

        ax2d.set_xlabel("X (nm)")
        ax2d.set_ylabel("Y (nm)")
        ax2d.set_title("2D projection of nucleosomes in slice")
        ax2d.set_aspect("equal", adjustable="box")

        half = self.width / 2.0
        ax2d.set_xlim(-half, half)
        ax2d.set_ylim(-half, half)

        plt.tight_layout()
        plt.show()

class STORMSimulator:
    def __init__(self, width=1000, resolution=10, slice_thickness=400, slice_z=0,
                 mean_antibodies_per_nucleosome=0.5, localization_precision=10, antibody_length=10,
                 mean_localizations_per_antibody=10, decay=True):
        self.width = width
        self.resolution = resolution
        self.slice_z = slice_z
        self.slice_thickness = slice_thickness
        self.localization_precision = localization_precision
        self.antibody_length = antibody_length
        self.mean_localizations_per_antibody = mean_localizations_per_antibody
        self.mean_antibodies_per_nucleosome = mean_antibodies_per_nucleosome
        self.decay = decay

    def slice_weights(self, positions):
        if self.decay:
            dz = np.abs(positions[:, 2] - self.slice_z)
            weights = np.exp(-0.5 * (dz / (self.slice_thickness / 2))**2)
            return weights
        weights = (positions[:, 2] > self.slice_z - self.slice_thickness / 2) & (positions[:, 2] < self.slice_z + self.slice_thickness / 2)
        return weights.astype(float)

    def antibody_counts(self, positions):
        # antibody_rate = 1.0 / (1 + self.mean_antibodies_per_nucleosome)
        # n_antibodies = np.random.geometric(antibody_rate, size=positions.shape[0]) - 1

        antibody_rate = self.mean_antibodies_per_nucleosome
        n_antibodies = np.random.poisson(antibody_rate, size=positions.shape[0])
        return n_antibodies

    def antibody_displacements(self, n_antibodies):
        antibody_displacements = np.random.rand(n_antibodies * 3).reshape(-1, 3)
        antibody_displacements -= 0.5
        antibody_displacements /= np.linalg.norm(antibody_displacements, axis=1)[:, None]
        antibody_displacements *= self.antibody_length
        return antibody_displacements

    def localization_counts(self, antibody_positions):
        localization_rate = 1.0 / (1 + self.mean_localizations_per_antibody)
        n_localizations = np.random.geometric(localization_rate, size=antibody_positions.shape[0]) - 1
        n_localizations = np.random.binomial(n_localizations, self.slice_weights(antibody_positions))
        return n_localizations

    def localization_displacements(self, n_localizations):
        localization_displacements = np.random.normal(0, self.localization_precision, size=(n_localizations, 3))
        return localization_displacements

    def get_localizations(self, positions):
        # Sample antibody positions
        n_antibodies = self.antibody_counts(positions)
        antibody_to_nucleosome = np.arange(positions.shape[0]).repeat(n_antibodies)
        antibody_displacements = self.antibody_displacements(len(antibody_to_nucleosome))
        antibody_positions = positions[antibody_to_nucleosome] + antibody_displacements

        # Sample localizations
        n_localizations = self.localization_counts(antibody_positions)
        localization_to_antibody = np.arange(antibody_to_nucleosome.shape[0]).repeat(n_localizations)
        localization_to_nucleosome = antibody_to_nucleosome[localization_to_antibody]
        localization_displacements = self.localization_displacements(len(localization_to_antibody))
        localizations = antibody_positions[localization_to_antibody] + localization_displacements

        localizations[:, 0] += self.width / 2.0
        localizations[:, 1] += self.width / 2.0

        return localizations, localization_to_nucleosome

    def get_image(self, localizations):
        """"
        Return a 2D histogram image of localizations.

        The localizations are binned into pixels of size `self.resolution` and the values
        are normalized by the pixel area to give a density.

        Parameters
        ----------
        localizations : (N, 2 or 3) array
            The (x, y) and maybe z coordinates of localizations.

        Returns
        -------
        h : 2D array
            Histogram image of localization density.
        """
        h = np.histogram2d(localizations[:, 0], localizations[:, 1],
                    bins=self.width//self.resolution,
                    range=[[0, self.width], [0, self.width]])[0]
        h /= self.resolution**2 # Convert to density
        return h

    def get_density_image(self, localizations, resolution=1, sigma=9):
        """
        Return an image representing the localization density.

        This should match the "density images" shown in Ricci et al. 2015:

        "The final images were rendered by representing each x-y position (localization)
        as a Gaussian with a width that corresponds to the determined localization precision (9 nm)."

        Parameters
        ----------
        localizations : (N, 2 or 3) array
            The (x, y) and maybe z coordinates of localizations.
        resolution : float
            Pixel size in nm.
        sigma : float
            Standard deviation of the Gaussian in nm. Default of 9 nm matches Ricci et al. 2015.
        Returns
        -------
        h : 2D array
            Histogram image of localization density.
        """
        h = np.histogram2d(localizations[:, 0], localizations[:, 1],
                    bins=self.width//resolution,
                    range=[[0, self.width], [0, self.width]])[0]
        h /= resolution**2 # Convert to density
        h = gaussian_filter(h, sigma=sigma/resolution)
        return h

    def plot_distributions(self):
        f, ax = plt.subplots(1, 3, figsize=(8, 2))
        positions = np.linspace(0, 500, 500)
        positions = np.array([np.zeros_like(positions), np.zeros_like(positions), positions]).T
        ax[0].plot(positions[:, 2], self.slice_weights(positions))
        ax[0].set_ylim(0)
        ax[0].set_xlabel('Z Position (nm)')
        ax[0].set_ylabel('Detection Probability')

        positions = np.zeros((1000, 3))
        antibodies = self.antibody_counts(positions)
        bins = np.arange(0, max(antibodies) + 2) - 0.5
        ax[1].hist(antibodies, bins=bins, density=True)
        ax[1].set_xlabel('Antibodies per Nucleosome')

        localizations = self.localization_counts(positions)
        bins = np.arange(0, max(localizations) + 2) - 0.5
        ax[2].hist(localizations, bins=bins, density=True)
        ax[2].set_xlabel('Localizations per Antibody')
        plt.show()

class ClutchAnalyzer:
    def __init__(self, filter_size=5, threshold=0.002, min_distance=2, resolution=10):
        self.filter_size = filter_size
        self.threshold = threshold
        self.min_distance = min_distance
        self.resolution = resolution
    
    def filter_density(self, h):
        """
        2D convolution with a square kernel.
        """
        kernel = np.ones((self.filter_size, self.filter_size), dtype=float)
        kernel /= kernel.sum()
        h_filtered = convolve2d(h, kernel, mode="same", boundary="fill", fillvalue=0)
        return h_filtered

    def discard_low_density_localizations(self, x, y, index, x_edges, y_edges, binary):
        """
        Remove localizations that fall on zero-valued pixels.
        """
        
        ix = np.searchsorted(x_edges, x, side="right") - 1
        iy = np.searchsorted(y_edges, y, side="right") - 1

        valid = (
            (ix >= 0) & (ix < binary.shape[1]) &
            (iy >= 0) & (iy < binary.shape[0])
        )

        ix = ix[valid]
        iy = iy[valid]
        index = index[valid]
        keep = binary[ix, iy] > 0

        return x[valid][keep], y[valid][keep], index[keep]

    def connected_components_4(self, binary):
        """
        Find 4-connected components in the binary image.
        """
        structure = np.array([[0, 1, 0],
                            [1, 1, 1],
                            [0, 1, 0]], dtype=int)
        labels, n_components = label(binary, structure=structure)
        return labels, n_components

    def find_initial_centroids(self, density_region):
        """
        Find local maxima in the density map for initialization.
        """
        peaks = peak_local_max(
            density_region,
            min_distance=self.min_distance,
            threshold_abs=0,
            threshold_rel=0,
            exclude_border=True
        )
        return peaks


    def kmeans_like_clustering(self, x, y, centroids, max_iter=100, tol=1e-3):
        """
        Distance-based clustering with centroid updates.

        centroids : array of shape (K, 2)
        """
        points = np.column_stack([x, y])
        K = centroids.shape[0]

        for _ in range(max_iter):
            # Assign points to nearest centroid
            d2 = np.sum((points[:, None, :] - centroids[None, :, :])**2, axis=2)
            labels = np.argmin(d2, axis=1)

            new_centroids = np.zeros_like(centroids)
            for k in range(K):
                members = points[labels == k]
                if members.size == 0:
                    new_centroids[k] = centroids[k]
                else:
                    new_centroids[k] = members.mean(axis=0)

            shift = np.sqrt(np.sum((new_centroids - centroids)**2))
            centroids = new_centroids

            if shift < tol:
                break

        return labels, centroids

    def run(self, localizations, h, localization_to_nucleosome, debug=False):
        # STORM data consisting in (x,y) localization lists were used to construct discrete localization images,
        # such that each pixel has a value equal to the number of localizations falling within the pixel
        # area (pixel size = 10 nm). From the localization images, density maps were obtained by 2-dimensional
        # convolution with a square kernel (5x5 pixels2). A constant threshold was used to digitize the density
        # maps into binary images, such that pixels have a value of 1 where the density is larger than the threshold
        # value and a value of 0 elsewhere.
        # ... The threshold value (0.002 nm−2) giving a ratio < 2x10−4 was used for image analysis.
        density = self.filter_density(h)
        binary = density > self.threshold

        x_edges = np.arange(0, binary.shape[1] + 1) * self.resolution
        y_edges = np.arange(0, binary.shape[0] + 1) * self.resolution
        x, y = localizations[:, 0], localizations[:, 1]

        # Localizations falling on zero-valued pixels of the binary images (low-density areas)
        # were discarded from further analysis.
        x_filt, y_filt, localization_to_nucleosome_filt = self.discard_low_density_localizations(x, y, localization_to_nucleosome, x_edges, y_edges, binary)

        if debug:
            f, ax = plt.subplots(1, 2, figsize=(10, 4))
            ax[0].set_title('Smoothed density')
            plot_density(density, ax=ax[0], resolution=self.resolution, colorbar=True)

            ax[1].set_title('Binary with locs (r: all, g: filtered)')
            plot_density(binary, ax=ax[1], resolution=self.resolution, colorbar=True)
            ax[1].scatter(x, y, c='red', s=1, alpha=1)
            ax[1].scatter(x_filt, y_filt, c='green', s=1, alpha=1)
            plt.show()

        # Connected components of the binary image, composed by adjacent non-zero pixels (4-connected neighbors),
        # were sequentially singled out and analyzed.
        labels_img, n_components = self.connected_components_4(binary)

        results = []
        for comp_id in range(1, n_components + 1):
            mask = labels_img == comp_id
            if not np.any(mask):
                continue

            density_region = density * mask

            # Localizations inside this connected component
            ix = np.searchsorted(x_edges, x_filt, side="right") - 1
            iy = np.searchsorted(y_edges, y_filt, side="right") - 1
            in_region = mask[ix, iy]

            x_region = x_filt[in_region]
            y_region = y_filt[in_region]
            localization_to_nucleosome_region = localization_to_nucleosome_filt[in_region]

            if x_region.size == 0:
                continue

            # Initialization values for the number of clusters and the relative centroid coordinates
            # were obtained from local maxima of the density map within the connected region,
            # calculated by means of a peak finding routine.
            peaks = self.find_initial_centroids(density_region)
            if peaks.shape[0] == 0:
                continue

            # Convert peaks to physical coordinates
            initial_centroids = []
            for px, py in peaks:
                cx = x_edges[px] + self.resolution / 2
                cy = y_edges[py] + self.resolution / 2
                initial_centroids.append([cx, cy])
            initial_centroids = np.array(initial_centroids)

            # f, ax = plt.subplots()
            # plot_density(density_region, ax=ax, resolution=self.resolution, colorbar=False)
            # ax.scatter(centroids[:, 0], centroids[:, 1], c='r', s=50, marker='x')

            # New cluster centroid coordinates were iteratively calculated as the
            # average of localization coordinates belonging to the same cluster.
            # The procedure was iterated until convergence of the sum of the squared
            # distances between localizations and the associated cluster and provided cluster
            # centroid positions and number of localizations per cluster.
            labels_pts, centroids = self.kmeans_like_clustering(
                x_region, y_region, initial_centroids
            )

            if debug:
                f, ax = plt.subplots(1, 2, figsize=(10, 5), sharex=True, sharey=True)
                ax[0].set_title(f'component {comp_id}: (g: initial, r: final)')
                plot_density(density_region, ax=ax[0], resolution=self.resolution, colorbar=False)
                ax[1].scatter(x_region, y_region, c=labels_pts, s=5, cmap='tab20')
                for a in ax:
                    a.scatter(initial_centroids[:, 0], initial_centroids[:, 1], c='g', s=50, marker='+')
                    a.scatter(centroids[:, 0], centroids[:, 1], c='r', s=20, marker='x')
                plt.show()

            # Cluster statistics
            for k in range(centroids.shape[0]):
                members = (labels_pts == k)
                if not np.any(members):
                    continue

                pts = np.column_stack([x_region[members], y_region[members]])
                nucleosomes = np.unique(localization_to_nucleosome_region[members])
                centroid = centroids[k]

                # Cluster sizes were calculated as the SD of localization
                # coordinates from the relative cluster centroid.
                sdx = np.sqrt(np.mean((pts[:, 0] - centroid[0])**2))
                sdy = np.sqrt(np.mean((pts[:, 1] - centroid[1])**2))
                area = np.pi * ((sdx + sdy) / 2)**2

                results.append({
                    "centroid_x": centroid[0],
                    "centroid_y": centroid[1],
                    "n_localizations": pts.shape[0],
                    "sdx": sdx,
                    "sdy": sdy,
                    "area": area,
                    'pts': pts,
                    'nucleosomes': nucleosomes,
                })
        return results, density, binary

class ChromSTEMAnalyzer:
    def __init__(self, resolution=2, filter_radius=5,
                 contrast_block_size=120, contrast_clip_limit=0.03,
                 peak_threshold=0.75, peak_min_distance=5,
                 rf_shift_range=5, rf_power_law_fit_max=40,
                 rf_deviation_from_fit=0.95, rf_n_samples=50,
                 rf_density_increase_factor=1.1,
                 radii_range=(30, 301), radii_n_points=200,
                 local_exponent_threshold=2.0, local_exponent_window=10,
                 mass_scaling_density='original'):
        self.resolution = resolution

        self.filter_radius = filter_radius

        self.contrast_block_size = contrast_block_size
        self.contrast_clip_limit = contrast_clip_limit

        self.peak_threshold = peak_threshold
        self.peak_min_distance = peak_min_distance

        self.rf_shift_range = rf_shift_range
        self.rf_power_law_fit_max = rf_power_law_fit_max
        self.rf_deviation_from_fit = rf_deviation_from_fit
        self.rf_n_samples = rf_n_samples
        self.rf_density_increase_factor = rf_density_increase_factor

        self.local_exponent_threshold = local_exponent_threshold
        self.local_exponent_window = local_exponent_window

        self.mass_scaling_density = mass_scaling_density

        self.radii = np.logspace(np.log10(radii_range[0]), np.log10(radii_range[1]), radii_n_points)

    def filter_density(self, h):
        h_filtered = gaussian_filter(h, sigma=self.filter_radius)
        return h_filtered

    def enhance_contrast(self, h_filtered):
        h_rescale = exposure.equalize_adapthist(h_filtered,
            kernel_size=(self.contrast_block_size, self.contrast_block_size),
            clip_limit=self.contrast_clip_limit,
            )
        return h_rescale

    def identify_peaks(self, h_contrast):
        coords = peak_local_max(
            h_contrast,
            threshold_abs=self.peak_threshold,
            exclude_border=True,
            min_distance=self.peak_min_distance
        )
        return coords

    def distance_from_center(self, center, shape):
        y, x = np.indices(shape)
        cy, cx = center
        r = np.sqrt((x - cx)**2 + (y - cy)**2)
        return r

    def mass(self, h, center, radius, r=None):
        if r is None:
            r = self.distance_from_center(center, h.shape)
        mask = r <= radius / self.resolution
        mass = h[mask].sum()
        return mass

    def mass_scaling(self, h, center):
        r = self.distance_from_center(center, h.shape)
        masses = []
        for radius in self.radii:
            mass = self.mass(h, center, radius, r=r)
            masses.append(mass)
        return np.array(masses)

    def jittered_mass_scaling(self, h, center):
        masses = np.zeros(self.radii.shape)
        for _ in range(self.rf_n_samples):
            shift_i = np.random.randint(-self.rf_shift_range, self.rf_shift_range)
            shift_j = np.random.randint(-self.rf_shift_range, self.rf_shift_range)
            _center = (center[0] + shift_i, center[1] + shift_j)
            masses += self.mass_scaling(h, _center)
        masses /= self.rf_n_samples
        return masses

    def local_exponent_criteria(self, local_exponents):
        decayed = False
        for i in range(len(local_exponents)):
            e = local_exponents[i]
            if np.isnan(e):
                pass
            elif decayed and e >= self.local_exponent_threshold:
                return i
            elif e < self.local_exponent_threshold:
                decayed = True
        return None

    def compute_rf(self, h, peak):
        masses = self.jittered_mass_scaling(h, peak)

        # Fit power law to inner region
        mask = self.radii <= self.rf_power_law_fit_max
        params = fit_power_law(self.radii[mask], masses[mask])
        fit_masses = power_law(self.radii, *params)

        # Criteria 1: deviation from power law fit
        deviates = masses < self.rf_deviation_from_fit * fit_masses

        # Criteria 2: density increases
        density = masses / (np.pi * self.radii**2)
        min_density = np.array([density[:i+1].min() for i in range(len(density))])
        deviates |= density > min_density * self.rf_density_increase_factor

        # Criteria 3: local power law exponent increases to more than 2.
        w = self.local_exponent_window
        local_exponents = np.full(masses.shape, np.nan)
        for i in range(w, len(masses)-w):
            local_params = fit_power_law(self.radii[i-w:i+w+1], masses[i-w:i+w+1])
            local_exponents[i] = local_params[1]

        local_exponent_deviation_index = self.local_exponent_criteria(local_exponents)
        if local_exponent_deviation_index is not None:
            deviates[local_exponent_deviation_index] = True

        deviation_index = np.argmax(deviates) # First index where deviation occurs
        return masses, local_exponents, params, deviation_index

    def run(self, h):
        h_filtered = self.filter_density(h)
        h_contrast = self.enhance_contrast(h_filtered)
        peaks = self.identify_peaks(h_contrast)

        if self.mass_scaling_density == 'filtered':
            h_for_rf = h_filtered
        elif self.mass_scaling_density == 'contrast':
            h_for_rf = h_contrast
        elif self.mass_scaling_density == 'original':
            h_for_rf = h
        else:
            raise ValueError("mass_scaling_density must be 'original', 'filtered', or 'contrast'.")

        rfs = np.zeros(len(peaks))
        all_masses = []
        all_params = []
        all_local_exponents = []
        cvcs = np.zeros(len(peaks))
        for peak_i in range(len(peaks)):
            peak = peaks[peak_i]
            masses, local_exponents, params, deviation_index = self.compute_rf(h_for_rf, peak)
            rfs[peak_i] = self.radii[deviation_index]
            cvcs[peak_i] = self.mass(h, peak, rfs[peak_i]) / self.mass(np.ones_like(h), peak, rfs[peak_i])
            all_masses.append(masses)
            all_params.append(params)
            all_local_exponents.append(local_exponents)


        results = {
            'peaks': peaks,
            'rfs': rfs,
            'radii': self.radii,
            'masses': all_masses,
            'local_exponents': all_local_exponents,
            'params': all_params,
            'cvcs': cvcs,
            'h': h,
            'h_filtered': h_filtered,
            'h_contrast': h_contrast,
        }
        return results
        

    def plot_density(self, h, ax=None, colorbar=False, **kwargs):
        plot_density(h, ax, self.resolution, colorbar=colorbar, **kwargs)
    
    def plot_peaks(self, peaks, rfs=None, ax=None, colors=None, s=5):
        if ax is None:
            ax = plt.gca()
        x = peaks[:, 0] * self.resolution
        y = peaks[:, 1] * self.resolution
        ax.scatter(x, y, c=colors if colors is not None else 'r', s=s)
        if rfs is not None:
            for _x, _y, rf in zip(x, y, rfs):
                ax.add_patch(plt.Circle((_x, _y), rf, color='w', fill=False, lw=0.25))

    def plot_densities(self, results, ax=None):
        if ax is None:
            f, ax = plt.subplots(1, 3, figsize=(7, 1.8))
        ax[0].set_title('Original Density', fontsize=8)
        self.plot_density(results['h'], ax[0])
        ax[1].set_title('Filtered Density', fontsize=8)
        self.plot_density(results['h_filtered'], ax[1])
        ax[2].set_title('Contrast Enhanced Density', fontsize=8)
        self.plot_density(results['h_contrast'], ax[2])
        #self.plot_peaks(results['peaks'], ax=ax[2])
        return f, ax

    def plot_domains(self, results, ax=None, s=5):
        if ax is None:
            fig, ax = plt.subplots(figsize=(6, 6))
        self.plot_density(results['h'], ax)
        self.plot_peaks(results['peaks'], results['rfs'], ax=ax, s=s)

    def plot_rf_fit(self, results, first_peak=0, max_peaks=5, add_space=False):
        if add_space:
            f, ax = plt.subplots(3, 1, figsize=(3, 4), sharex=True, gridspec_kw={'hspace': 0.5})
        else:
            f, ax = plt.subplots(3, 1, figsize=(3, 4), sharex=True)
        for peak_i in range(first_peak, min(first_peak + max_peaks, len(results['peaks']))):
            c = 'C'+str(peak_i)
            masses = results['masses'][peak_i]
            params = results['params'][peak_i]
            density = masses / (np.pi * results['radii']**2)
            fit_masses = power_law(results['radii'], *params)
        
            lw = 1
            ax[0].plot(results['radii'], masses, c=c, lw=lw)
            ax[0].plot(results['radii'], fit_masses, ls='--', c=c, lw=lw)
            ax[1].plot(results['radii'], density, c=c)
            ax[2].plot(results['radii'], 1+results['local_exponents'][peak_i], c=c, lw=lw)

            for a in ax:
                a.axvline(results['rfs'][peak_i], c=c, ls=':', lw=lw)
    
        ax[0].set_xscale('log')
        ax[0].set_yscale('log')
        ax[1].set_xscale('log')

        ax[0].set_ylabel('Mass')
        ax[1].set_ylabel('Density')
        ax[2].set_ylabel('Exponent')
        ax[2].set_xlabel('Radius (nm)')
        return f, ax
