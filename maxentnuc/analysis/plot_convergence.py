from .mei_analyzer import MEIAnalyzer
import click


@click.command()
@click.argument('config', default='./config.yaml')
@click.option('--scale', default=0.1)
@click.option('--stride', default=5)
def main(config, scale, stride):
    analyzer = MEIAnalyzer(config, scale=scale)
    analyzer.compare_maps_convergence()
    analyzer.marginal_convergence(stride=stride)
    analyzer.end_to_end_distance_convergence(skip=1, plot_freq=-1)


if __name__ == '__main__':
    main()
