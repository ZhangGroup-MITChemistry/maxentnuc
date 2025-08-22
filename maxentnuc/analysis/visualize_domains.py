import numpy as np
import os
import click
from maxentnuc.analysis.mei_analyzer import MEIAnalyzer
from maxentnuc.analysis.domain_analyzer import DomainAnalyzer
from maxentnuc.analysis.insulation import get_segments, get_insulation_scores


def change_paths(fname, old, new):
    if new is not None:
        cmd = f"sed -i 's#{old}#{new}#g' {fname}"
        print(cmd)
        os.system(cmd)


def labels_to_colors(labels, ncolors=9):
    noise = labels == -1
    colors = labels % ncolors + 1
    colors[noise] = -1
    return colors


@click.command()
@click.argument('data-root')
@click.option('--visualization-root', default=None, help='Path to be used when visualizing. Useful if running this script on a cluster and VMD locally.')
@click.option('--skip', default=1100)
def main(data_root, visualization_root, skip):
    prod = {
        'nanog': {'config': f'{data_root}/mei_runs/nanog/v36/config.yaml', 'iteration': 18},
        'klf1': {'config': f'{data_root}/mei_runs/klf1/v3/config.yaml', 'iteration': 11},
        'ppm1g': {'config': f'{data_root}/mei_runs/ppm1g/v2/config.yaml', 'iteration': 10},
        'sox2': {'config': f'{data_root}/mei_runs/sox2/v11/config.yaml', 'iteration': 13},
        'fbn2' : {'config': f'{data_root}/mei_runs/fbn2/v2/config.yaml', 'iteration': 12},
    }

    for name in prod:
        print(f'Processing {name}: {prod[name]['iteration']}...')
        prod[name]['mei'] = MEIAnalyzer(prod[name]['config'], scale=0.1)
        trajectory = prod[name]['mei'].get_positions(prod[name]['iteration'], skip=skip, burnin=0)
        prod[name]['trajectory'] = trajectory.reshape(-1, *trajectory.shape[-2:])

    # Chain.
    for name in prod:
        colors = np.array([10 * np.arange(len(positions)) / len(positions) for positions in prod[name]['trajectory']])

        prod[name]['mei'].visualize(iteration=prod[name]['iteration'], script_base_name=f'{name}_chain', skip=skip,
                                    burnin=0, colors=colors, sele='name NUC')
        change_paths(f"{name}_chain.tcl", data_root, visualization_root)

    # DBSCAN.
    analyzer = DomainAnalyzer(30, 200)
    domains = {}
    for name, info in prod.items():
        domains[name] = analyzer.analyze_trajectory(info['trajectory'])

    for name in prod:
        labels = np.array([d.labels for d in domains[name]])
        colors = labels_to_colors(labels)

        prod[name]['mei'].visualize(iteration=prod[name]['iteration'], script_base_name=f'{name}_dbscan', skip=1100,
                                    burnin=0, colors=colors, sele='name NUC')
        change_paths(f"{name}_dbscan.tcl", data_root, visualization_root)

    # Insulation domains.
    for name in prod:
        insulation_scores = get_insulation_scores(prod[name]['trajectory'], windows=[10])[10]
        insulation_scores[np.isinf(insulation_scores)] = np.nanmin(insulation_scores)
        mask = insulation_scores > -2.5
        labels = np.array([get_segments(m) for m in mask])
        colors = labels_to_colors(labels)

        prod[name]['mei'].visualize(iteration=prod[name]['iteration'], script_base_name=f'{name}_ins', skip=1100,
                                    burnin=0, colors=colors, sele='name NUC')
        change_paths(f"{name}_ins.tcl", data_root, visualization_root)


if __name__ == '__main__':
    main()
