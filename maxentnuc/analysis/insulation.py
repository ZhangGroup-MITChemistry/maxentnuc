from cooltools.api.insulation import _insul_diamond_dense
from cooltools.lib.peaks import find_peak_prominence
from skimage.filters import threshold_li
from maxentnuc.analysis.mei_analyzer import MEIAnalyzer
from .analysis import *
import click
import pickle
import warnings


def get_insulation_scores(trajectory, windows, contact_threshold=40, ignore_diags=2):
    """
    Compute the single-molecule insulation scores for a given trajectory.
    """
    insulation_scores = {}
    for window in windows:
        insulation_scores[window] = np.zeros(trajectory.shape[:2])

    for t, positions in enumerate(trajectory):
        contact_map = (get_distance_matrix(positions) < contact_threshold).astype(float)
        for window in windows:
            with warnings.catch_warnings(): # No contacts gives -inf, this is fine.
                scores = np.log2(_insul_diamond_dense(contact_map, window=window, norm_by_median=False,
                                                      ignore_diags=ignore_diags))

            insulation_scores[window][t] = scores
    return insulation_scores


def mask_insulation_scores(insulation_scores, tol=1e-2):
    """
    Set scores that are the same in consecutive frames to NaN.
    """
    masked_insulation_scores = insulation_scores.copy()
    for i in range(insulation_scores.shape[1] - 1):
        mask = np.abs(insulation_scores[:, i] - insulation_scores[:, i + 1]) < tol
        masked_insulation_scores[mask, i + 1] = np.nan
    return masked_insulation_scores


def get_boundaries(insulation_scores, threshold='Li'):
    """
    Find strong boundaries in insulation scores.
    """
    boundaries = []
    for scores in insulation_scores:
        poss, proms = find_peak_prominence(-scores)
        if threshold == 'Li':
            mask = proms > threshold_li(proms)
        elif threshold is int or float:
            mask = proms > threshold
        else:
            raise ValueError(f'{threshold}')
        strong_boundaries = poss[mask]
        boundaries += [strong_boundaries]
    return boundaries


def get_peak_prominences(insulation_scores):
    """
    Get an unfiltered track of peak prominences.
    """
    peaks = np.zeros(insulation_scores.shape)
    for t, scores in enumerate(insulation_scores):
        poss, proms = find_peak_prominence(-scores)
        peaks[t, poss] = proms
    return peaks


def get_sections(n, boundaries):
    """
    Get the sections between the boundaries.
    """
    sections = np.zeros((len(boundaries), n))
    for t in range(len(boundaries)):
        for i in range(len(boundaries[t]) - 1):
            sections[t, boundaries[t][i]:boundaries[t][i + 1]] = i + 1
        if len(boundaries[t]) > 0:
            sections[t, boundaries[t][-1]:] = len(boundaries[t]) + 1
    return sections


def get_segments(mask):
    """
    Given a mask, return segments with True values.

    Example:
        mask = [False, True, True, False, False, True, True, True, False]
        segments = [-1, 0, 0, -1, -1, 1, 1, 1, -1]
    """
    segments = np.zeros(mask.shape, dtype=int) - 1
    segment = -1
    in_segment = False
    for i in range(mask.shape[0]):
        if mask[i]:
            if not in_segment:
                segment += 1
                in_segment = True
            segments[i] = segment
        else:
            in_segment = False
    return segments


def get_partition(track, threshold=0, window=0):
    """
    Decompose a track into domains and linkers.

    A break between domains (a linker) is introduced where the track is less than the threshold. Where runs
    of more than 1 value greater than the threshold are found the linker is adjusted based on the provided window.
    It is assumed that the track is computed considering a window of size 2 * window + 1. Therefore, it makes sense
    to include up to window values on the end of each linker in the corresponding domain. Consider the case where
    these are radius of gyration values or insulation scores.
    """

    # Get linkers, defined as values greater than threshold.
    mask = track > threshold
    linkers = []
    linker_start = None
    for i in range(len(mask)):
        if mask[i]:
            if linker_start is None:
                linker_start = i
        elif linker_start is not None:
            linkers += [(linker_start, i)]
            linker_start = None

    # Adjust the sections based on the window.
    adjusted_linkers = []
    for linker in linkers:
        start, end = linker

        peak_height = track[start:end].max()
        peaks = np.where(track[start:end] == peak_height)[0]
        peak = start + peaks[len(peaks) // 2]

        start = min(peak, start + window)
        end = max(peak+1, end - window)
        assert end > start
        adjusted_linkers += [(start, end)]

    # Convert to a track.
    if not adjusted_linkers:
        return np.zeros(len(track), dtype=int)

    labels = np.full(len(track), -1, dtype=int)
    labels[:adjusted_linkers[0][0]] = 0
    for i in range(len(adjusted_linkers) - 1):
        labels[adjusted_linkers[i][1]:adjusted_linkers[i+1][0]] = i + 1
    labels[adjusted_linkers[-1][1]:] = len(adjusted_linkers)
    return labels


def get_regions_as_track(n, regions):
    tracks = np.zeros((len(regions), n))
    for t in range(len(regions)):
        for s, e in regions[t]:
            tracks[t, s:e] = 1
    return tracks


def boundary_probabilities(n, boundaries):
    p = np.zeros(n)
    for boundary in boundaries:
        p[boundary] += 1
    return p / len(boundaries)


@click.command()
@click.option('--radius', default=40)
@click.option('--ignore-diags', default=2)
@click.option('--config', default='config.yaml')
@click.option('--scale', default=0.1)
@click.option('--skip', default=1)
@click.option('--iteration', default=-1)
def main(radius, ignore_diags, config, scale, skip, iteration):
    mei = MEIAnalyzer(config, scale=scale)

    if iteration == -1:
        iteration = mei.get_iterations()[-1]

    trajectory = mei.get_positions(iteration, skip=skip)
    trajectory = trajectory.reshape(-1, *trajectory.shape[2:])

    windows = [5, 10, 30, 50, 100, 250, 500]
    insulation_scores = get_insulation_scores(trajectory, windows=windows, contact_threshold=radius,
                                              ignore_diags=ignore_diags)

    out = {}
    out['scores'] = insulation_scores
    out['params'] = {'radius': radius, 'ignore_diags': ignore_diags}

    with open('insulation.pkl', 'wb') as fp:
        pickle.dump(out, fp)


if __name__ == '__main__':
    """
    cd ~/mei_runs/fbn2/v2
    sbatch --wrap="python -m na_genes.analysis.insulation --skip 11 --iteration 12"
    """
    main()
