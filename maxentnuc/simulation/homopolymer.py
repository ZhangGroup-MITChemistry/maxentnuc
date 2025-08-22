"""
Runs a

#!/bin/bash
#SBATCH -J homopolymer
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=24:00:00

python -m maxentnuc.simulation.homopolymer config.yaml test
"""
import yaml
import click
import openmm
from .model import PolymerModel, write_psf, simulate


@click.command()
@click.argument('config_fname')
@click.argument('basename')
@click.argument('replicate', type=int)
def main(config_fname, basename, replicate):
    with open(config_fname) as fp:
        config = yaml.safe_load(fp)

    psf = f'{basename}_topology.{replicate}.psf'
    dcd = f'{basename}_trajectory.{replicate}.dcd'
    log = f'{basename}_trajectory.{replicate}.log'

    chrom, startend = config['region'].split(':')
    start, end = startend.split('-')
    start = int(start.replace(',', ''))
    end = int(end.replace(',', ''))
    N = (end - start) // config['resolution']

    print('Building model...', flush=True)
    model = PolymerModel(N, **config['model_params'])
    model.build()
    print('Built model', flush=True)
    write_psf(model.topology, psf)
    print('Wrote psf', flush=True)
    positions = model.generate_initial_positions()
    print('Generated initial positions', flush=True)
    simulate(topology=model.topology, system=model.system, positions=positions,
             dt=config['dt'] * openmm.unit.picoseconds,
             friction=config['friction'] / openmm.unit.picoseconds,
             T=None,
             n_steps=config['n_steps'],
             report_interval=config['report_interval'],
             dcd=dcd,
             log=log,
             platform=config['platform'],
             verbose=True
             )


if __name__ == '__main__':
    main()
