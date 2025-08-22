"""
Interface with the neighbor_balance repo.
"""

import numpy as np
from neighbor_balance.ice import ice_balance_with_interpolation, get_capture_rates
from neighbor_balance.neighbor import normalize_contact_map_neighbor, normalize_contact_map_average
from neighbor_balance.smoothing import smooth_diagonals, interpolate_diagonals, coarsen_contact_map
from neighbor_balance.plotting import parse_region, ContactMap


def load_and_process_contact_map(contact_map_path, region, resolution,
                                 normalization, normalization_params,
                                 smoothing=None, smoothing_params=None,
                                 map_region=None, min_alpha=-1, capture_probes=None):
    """
    Load and normalize the contact map according to the configuration.
    """
    if normalization == 'per_loci':
        print('Warning: `per_loci` normalization is deprecated. Use `neighbor` instead.', flush=True)

    # If map_region is specified, use that region for normalization and later we will extract the region of interest.
    if map_region is None:
        map_region = region

    if normalization == 'none':
        contacts = ContactMap.from_cooler(contact_map_path, resolution, map_region, min_alpha=min_alpha).contact_map
    elif normalization == 'average_neighbor':
        contacts = ContactMap.from_cooler(contact_map_path, resolution, map_region, min_alpha=min_alpha).contact_map
        contacts = normalize_contact_map_average(contacts, **normalization_params)
    elif normalization in ['per_loci', 'neighbor']:
        contacts = ContactMap.from_cooler(contact_map_path, resolution, map_region, min_alpha=min_alpha).contact_map
        contacts = normalize_contact_map_neighbor(contacts, **normalization_params)
    elif normalization == 'mask':
        contacts = ContactMap.from_cooler(contact_map_path, resolution, map_region, balance=False).contact_map
        chrom, start, end = parse_region(map_region)
        if capture_probes is not None:
            capture_rates = get_capture_rates(capture_probes, chrom, start, end, bin_size=resolution)
        else:
            capture_rates = None

        contacts = ice_balance_with_interpolation(contacts, capture_rates=capture_rates, **normalization_params['ice'])

        if normalization_params['neighbor']:
            contacts = normalize_contact_map_neighbor(contacts, **normalization_params['normalization'])
        else:
            contacts = normalize_contact_map_average(contacts, **normalization_params['normalization'])

    else:
        raise ValueError(f"Invalid normalization method {normalization}")

    if smoothing is not None:
        contacts = smooth_contact_map(contacts, smoothing, smoothing_params)

    # Extract the region of interest and make sure all the coordinates are aligned.
    chrom, start, end = parse_region(region)
    map_chrom, map_start, map_end = parse_region(map_region)

    assert chrom == map_chrom, 'Chromosomes do not match.'
    assert start >= map_start, 'Region start is before map start.'
    assert end <= map_end, 'Region end is after map end.'
    assert (start - map_start) % resolution == 0, 'Region start is not aligned with map start.'
    assert (end - start) % resolution == 0, 'Region size is not a multiple of the bead size.'
    assert (map_end - map_start) % resolution == 0, 'Map region size is not a multiple of the bead size.'
    assert start % resolution == 0, 'Region start is not aligned with the bead size.'
    assert end % resolution == 0, 'Region end is not aligned with the bead size.'
    assert contacts.shape[0] == (map_end - map_start) // resolution, 'Contact map size does not match region.'

    start_offset = (start - map_start) // resolution
    end_offset = (end - map_start) // resolution
    return contacts[start_offset:end_offset, start_offset:end_offset]


def smooth_contact_map(contact_map, smoothing, smoothing_args):
    """
    Smooths the contact map using the specified method and arguments.

    Parameters
    ----------
    contact_map: np.ndarray
        The contact map to be smoothed.
    smoothing: str
        The method used for smoothing. Currently only 'diagonal' is supported.
    smoothing_args: dict
        The key-word arguments to be passed to the smoothing function.
    """
    if smoothing == 'diagonal':
        return smooth_diagonals(contact_map, **smoothing_args)
    elif smoothing == 'interpolate_diagonals':
        return interpolate_diagonals(contact_map, **smoothing_args)
    elif smoothing == 'coarsen':
        return coarsen_contact_map(contact_map, **smoothing_args)
    else:
        raise ValueError(f'Unknown smoothing method: {smoothing}')
