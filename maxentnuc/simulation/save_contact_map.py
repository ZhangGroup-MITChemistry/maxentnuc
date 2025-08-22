import click
from functools import partial
import numpy as np
from . import simulated_contacts as scm
from .mei import yaml_load, parse_config
from glob import glob


@click.command()
@click.argument('config', type=click.Path(exists=True))
@click.argument('base')
def main(config, base):
    config = parse_config(yaml_load(config))

    indicator = partial(scm.contact_indicator, tanh_sigma=config['contact_scale'], tanh_r_c=config['contact_r_c'])
    psf = f'{base}_topology.0.psf'
    dcds = sorted(glob(f'{base}_trajectory.*.dcd'))

    contacts = []
    for dcd in dcds:
        contacts += [scm.get_contacts(psf, dcd, indicator,
                                      burnin=config['n_steps_burnin'],
                                      atom_selection=config['atom_selection'],
                                      competition=config['competition'])]
    contacts = np.vstack(contacts)
    np.save(f'{base}_contacts.npy', contacts)


if __name__ == '__main__':
    main()
