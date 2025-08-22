from MDAnalysis import Universe
import topoly
import click


def topology(psf, dcd, atom_selection='name NUC CAP', stride=100, max_cross=100):
    out = []
    u = Universe(psf, dcd)
    for _ in u.trajectory[::stride]:
        struct = u.select_atoms(atom_selection).positions
        struct = [list(xyz) for xyz in struct]
        try:
            top = topoly.conway(struct, closure=topoly.Closure.MASS_CENTER, max_cross=max_cross)
        except ValueError:
            top = 'failed'

        print(top)
        out += [top]
    return out


@click.command()
@click.argument('psf')
@click.argument('dcd')
@click.option('--atom_selection', default='name NUC CAP')
@click.option('--stride', default=100)
@click.option('--max_cross', default=100)
def main(psf, dcd, **kwargs):
    topology(psf, dcd, **kwargs)


if __name__ == '__main__':
    main()
