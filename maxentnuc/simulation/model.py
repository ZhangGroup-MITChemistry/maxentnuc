import openmm
from openmm.app import Topology, Simulation, StateDataReporter, DCDReporter
from openmm.app.element import Element
import math
import numpy as np
import hilbert
import parmed
from openmm import unit
from .upper_triangular import triu_to_full
import os


def write_psf(top: Topology, fname: str):
    try:
        parmed.openmm.load_topology(top).save(fname, overwrite=True)
    except RecursionError:
        print(f'Did not save entire psf file: check number of beads {top.getNumAtoms()}')


def get_simulation_platform_and_properties(platform_name):
    if platform_name == 'CUDA':
        properties = {'Precision': 'mixed'}
    elif platform_name == 'OpenCL':
        properties = {'Precision': 'mixed'}
    elif platform_name == 'Reference':
        properties = {}
    elif platform_name == 'CPU':
        properties = {}
    else:
        assert False, platform_name

    platform = openmm.Platform.getPlatformByName(platform_name)
    return platform, properties


def simulate(*, topology, system, positions, dt, friction, platform,
             dcd, log, report_interval, n_steps, device_index=None, append=False, T=None, verbose=False) -> bool:
    """
    Run the simulation and return True if it completed successfully.
    """
    if T is None:
        T = 1 * unit.kilojoule_per_mole / (unit.BOLTZMANN_CONSTANT_kB * unit.AVOGADRO_CONSTANT_NA)  # kT = 1 kJ/mol
    integrator = openmm.LangevinMiddleIntegrator(T, friction, dt)

    platform, properties = get_simulation_platform_and_properties(platform)
    if device_index is not None:
        properties['DeviceIndex'] = device_index

    simulation = Simulation(topology, system, integrator, platform, properties)
    if verbose:
        print('Created simulation', flush=True)
    simulation.context.setPositions(positions)
    if verbose:
        print('Set positions', flush=True)
    # Lowering the tolerance from the default of 10 kJ/mol/nm is needed because the length scales of the simulation
    # are larger than typical. Failure to do this will result in the simulation crashing immediately.
    # The enlarged terminal nucleosomes seem to be the most problematic.
    # TODO: Do I need to minimize in later rounds? The system should already be in a reasonable configuration.
    simulation.minimizeEnergy(tolerance=0.1*unit.kilojoules_per_mole/unit.nanometer)

    if verbose:
        print('Minimized energy', flush=True)

    if append:
        assert os.path.exists(dcd) and os.path.exists(log)
    else:
        assert not os.path.exists(dcd) and not os.path.exists(log)

    if openmm.__version__ >= '8.0':
        simulation.reporters.append(
            DCDReporter(dcd, report_interval, append=append))
        simulation.reporters.append(
            StateDataReporter(log, report_interval,
                              potentialEnergy=True, kineticEnergy=True, totalEnergy=True, temperature=True,
                              step=True, time=True, progress=True, remainingTime=True, speed=True,
                              totalSteps=n_steps, append=append))
    else:
        # The append argument was added in OpenMM 8.0. I can only install OpenMM 7 on satori. I'm not sure if I will
        # edit mei.py so that it can work with append=False.
        assert not append
        simulation.reporters.append(
            DCDReporter(dcd, report_interval))
        simulation.reporters.append(
            StateDataReporter(log, report_interval,
                              potentialEnergy=True, kineticEnergy=True, totalEnergy=True, temperature=True,
                              step=True, time=True, progress=True, remainingTime=True, speed=True,
                              totalSteps=n_steps))

    try:
        simulation.step(n_steps)
        if verbose:
            print('Simulation completed', flush=True)
        return True
    except (ValueError, openmm.OpenMMException) as e:
        print(e)
        return False


class PolymerModel:
    """
    A simple model of chromatin consisting of spherical nucleosomes connected by class2 bonds.

    To maintain a knot free structure, DNA beads must be added between the nucleosomes to prevent strand crossing and
    enlarged terminal nucleosomes must be added to prevent knotting at the ends. These additions leave the chain
    kinetically trapped in an unknotted state.

    DNA particles are added as virtual sites to prevent strand crossing, the radius and number of DNA beads should be
    chosen such that the DNA beads are touching when the bond length is at its highest feasible value:
        max_bond_length - nucleosome_diameter < n_dna * dna_diameter
    ex: max_bond_length = 35, nucleosome_diameter = 11, dna_diameter = 5 -> n_dna > 4.8

    The end nucleosomes are enlarged to prevent knotting from the ends. To prevent this from overly biasing the system,
    cap nucleosomes are added between the terminal nucleosomes and the actual system.

    An attractive potential can be added between all nucleosome pairs by adjusting the `nuc_nuc_attraction`,
    `nuc_nuc_rc`, and `nuc_nuc_scale` parameters. This potential is a tanh potential that is applied uniformly to all
    nucleosome pairs. This is meant to model the inherent attraction between nucleosomes. In the future, this could be
    replaced with a more sophisticated potential that depends on the angle between nucleosomes.

    A non-uniform attractive potential can be added using the `add_tanh_potential` method. This allows different energy
    scales for each nucleosome pair and is used to make the system to match an experimental contact map.

    It is expected that the model be simulated with kT = 1. In OpenMM, we cannot set Boltzmann's constant to 1, so
    instead we set the temperature to 1 kJ/mol / (k*NA) ~ 120.3 K, which yields kT = 1. This is equivalent to using T=1
    and k=1, since the two are always multiplied.

    See the below links for more information on units:
    http://docs.openmm.org/latest/userguide/theory/01_introduction.html#units
    https://docs.lammps.org/units.html

    Parameters
    ----------
    n : int
        The number of nucleosomes in the chain.
    nucleosome_diameter : float
        The diameter of the nucleosomes.
    bond_length : float
        The equilibrium bond length of the class2 bonds.
    angle_constant : float
        The force constant for the angle potential.
    n_dna : int
        The number of DNA particles to add between each pair of nucleosomes.
    dna_diameter : float
        The diameter of the DNA particles.
    n_cap : int
        The number of nucleosomes to add to the ends of the chain.
    cap_nuc : float
        The weight of the tanh potential between nucleosomes and cap nucleosomes. Recommended to be 0.0.
    cap_cap : float
        The weight of the tanh potential between cap nucleosomes. Recommended to be 0.0.
    bond_length_fluctuation : float
        The factor by which to extend the bond length to account for fluctuations when adding DNA virtual sites.
    bond_k2 : float
        The second order force constant for the class2 bond potential.
    bond_k3 : float
        The third order force constant for the class2 bond potential.
    bond_k4 : float
        The fourth order force constant for the class2 bond potential.
    terminal_nucleosome_diameter : float
        If not None, increase the diameter of the terminal nucleosomes to prevent knotting.
    nuc_nuc_attraction : float
        Energy scale for a tanh potential applied uniformly to all nucleosome pairs.
    nuc_nuc_rc : float
        The cutoff distance for the tanh potential applied uniformly between nucleosomes.
    nuc_nuc_scale : float
        The scale factor for the tanh potential applied uniformly between nucleosomes.
    cap_nuc_attraction : float
        Energy scale for a tanh potential applied between cap nucleosomes and nucleosomes.
    cap_cap_attraction : float
        Energy scale for a tanh potential applied between cap nucleosomes.

    Attributes
    ----------
    n_total : int
        The total number of particles in the chain.
    topology : Topology
        The topology of the polymer chain.
    system : openmm.System
        The OpenMM System object.
    """
    def __init__(self, n, nucleosome_diameter=10, bond_length=24, angle_constant=0.0,
                 n_dna=0, dna_diameter=5, n_cap=0, cap_nuc=0.0, cap_cap=0.0, bond_length_fluctuation=1.5,
                 bond_k2=1/20, bond_k3=None, bond_k4=None, terminal_nucleosome_diameter=None,
                 nuc_nuc_attraction=0.0, nuc_nuc_rc=15.0, nuc_nuc_scale=1/3,
                 cap_nuc_attraction=0.0, cap_cap_attraction=0.0):
        self.n = n
        self.n_cap = n_cap
        self.n_total = n + 2*n_cap
        self.cap_nuc = cap_nuc
        self.cap_cap = cap_cap
        self.nucleosome_diameter = nucleosome_diameter
        self.bond_length = bond_length
        self.angle_constant = angle_constant
        self.n_dna = n_dna
        self.dna_diameter = dna_diameter
        self.bond_length_fluctuation = bond_length_fluctuation
        self.bond_k2 = bond_k2
        self.bond_k3 = bond_k3
        self.bond_k4 = bond_k4
        self.terminal_nucleosome_diameter = terminal_nucleosome_diameter
        self.nuc_nuc_attraction = nuc_nuc_attraction
        self.nuc_nuc_rc = nuc_nuc_rc
        self.nuc_nuc_scale = nuc_nuc_scale
        self.cap_nuc_attraction = cap_nuc_attraction
        self.cap_cap_attraction = cap_cap_attraction

        self.topology = Topology()
        self.system = openmm.System()

    def build(self):
        """
        Define the topology, system, and forces for the polymer chain.
        """
        self.define_topology()
        self.define_system()
        self.add_standard_forces()

    @staticmethod
    def create_element(symbol, mass):
        try:
            return Element.getBySymbol(symbol)
        except KeyError:
            return Element(0, symbol, symbol, mass)

    def define_topology(self):
        """
        Define the topology of the polymer chain. Each monomer is the same, so this is quite simple.
        """
        self.topology = Topology()

        element = self.create_element('N0', 1 * unit.atomic_mass_unit)
        chain = self.topology.addChain(id='Ch')
        atoms = []
        for i in range(self.n_cap):
            atoms += [self.topology.addAtom(f'CAP', element, self.topology.addResidue('CAP', chain))]
        for i in range(self.n):
            atoms += [self.topology.addAtom(f'NUC', element, self.topology.addResidue('NUC', chain))]
        for i in range(self.n_cap):
            atoms += [self.topology.addAtom(f'CAP', element, self.topology.addResidue('CAP', chain))]

        for a1, a2 in zip(atoms[:-1], atoms[1:]):
            self.topology.addBond(a1, a2)

        # Add zero mass beads to represent the DNA as virtual sites.
        dna_element = self.create_element('D0', 0 * unit.atomic_mass_unit)
        for _ in range(self.n_dna*(self.n_total - 1)):
            self.topology.addAtom(f'DNA', dna_element, self.topology.addResidue('DNA', chain))

    def generate_initial_positions(self):
        """
        Generate initial positions for the polymer chain.

        The nucleosomes are placed on a Hilbert curve, which has a fractal structure with no knots.
        """
        d = 3
        positions = hilbert.decode(np.arange(self.n_total), d,
                                   math.ceil(1/d * np.log2(self.n_total))).astype(int)
        assert (np.sum(np.abs(np.diff(positions, axis=0)), axis=1) == 1).all()
        for i in range(len(positions)):
            for j in range(i + 1, len(positions)):
                assert not np.all(np.isclose(positions[i], positions[j]))
        positions -= positions.mean(axis=0, dtype=int)

        positions = positions * self.bond_length
        if self.n_dna > 0:
            positions = np.vstack([positions, np.zeros((self.n_dna * (self.n_total - 1), 3))])
        return positions

    def define_system(self):
        """
        Define the OpenMM System. This must be done after the topology is defined and before the forces are added.
        """
        self.system = openmm.System()

        for atom in self.topology.atoms():
            self.system.addParticle(atom.element.mass)

        for i in range(self.n_total - 1):
            for j in range(self.n_dna):
                # Consider an extended bond length to account for fluctuations.
                extended_bond_length = self.bond_length_fluctuation * self.bond_length
                sep = extended_bond_length - self.nucleosome_diameter
                w = (self.nucleosome_diameter / 2 + (2*j + 1) * sep / (2*self.n_dna)) / extended_bond_length
                vs = openmm.TwoParticleAverageSite(i, i + 1, w, 1-w)
                self.system.setVirtualSite(self.n_total + i*self.n_dna + j, vs)

        self.system.addForce(openmm.CMMotionRemover())

    def add_standard_forces(self):
        """
        Add the standard forces to the system. This includes the hard-core potential, the class2 bond potential, and
        (optionally) the angle potential.
        """
        self.add_hard_core_potential()
        self.add_class2_bond_potential()
        if self.terminal_nucleosome_diameter is not None:
            self.add_terminal_potential()
        if self.angle_constant > 0.0:
            self.add_angle_potential()
        if self.nuc_nuc_attraction != 0.0:
            self.add_uniform_tanh_potential()

    def add_class2_bond_potential(self):
        """
        Add the class2 bond potential to the system.
        """
        bond_k3 = self.bond_k2 ** 2 if self.bond_k3 is None else self.bond_k3
        bond_k4 = self.bond_k2 ** 3 if self.bond_k4 is None else self.bond_k4
        force = openmm.CustomBondForce(f'{self.bond_k2} * (r - {self.bond_length})^2 '
                                       f'+ {bond_k3} * (r - {self.bond_length})^3'
                                       f'+ {bond_k4} * (r - {self.bond_length})^4')
        for i in range(self.n_total - 1):
            force.addBond(i, i + 1)
        force.setUsesPeriodicBoundaryConditions(False)
        self.system.addForce(force)

    def add_angle_potential(self):
        """
        Add an angle potential encoding the tendency for nucleosomes to induce a 180 degree bend in the DNA.
        """
        force = openmm.HarmonicAngleForce()
        for i in range(self.n_total - 2):
            force.addAngle(i, i + 1, i + 2, 0.0, self.angle_constant)
        force.setUsesPeriodicBoundaryConditions(False)
        self.system.addForce(force)

    def add_exclusions(self, force):
        """
        Add exclusions to a nonbonded force.

        Interactions are excluded between DNA particles and neighboring nucleosomes and pairs of DNA particles that
        are "bonded" to the same nucleosome. These exclusions allow the string of DNA beads between two nucleosomes
        to overlap and thereby not affect the bond length and to prevent the DNA beads from getting in the way of
        acute angles between nucleosomes.

        Note: I previously excluded interactions between neighboring nucleosomes, but I know believe this is incorrect.
        Neighboring nucleosomes should not be allowed to overlap.
        """
        for i in range(self.n_total-1):
            for j in range(self.n_dna):
                dna_bead = self.n_total + i*self.n_dna + j
                # Add neighboring nucleosome beads to the exclusion list.
                first_nucleosome_bead = i
                second_nucleosome_bead = i + 1
                force.addExclusion(first_nucleosome_bead, dna_bead)
                force.addExclusion(second_nucleosome_bead, dna_bead)
                # Add DNA beads between the nucleosomes to the exclusion list.
                for k in range(j+1, self.n_dna):
                    other_dna_bead = self.n_total + i * self.n_dna + k
                    force.addExclusion(dna_bead, other_dna_bead)
                # Add DNA beads to the next nucleosome to the exclusion list.
                if i < self.n_total - 2:
                    for k in range(0, self.n_dna):
                        other_dna_bead = self.n_total + (i + 1) * self.n_dna + k
                        force.addExclusion(dna_bead, other_dna_bead)

    def add_hard_core_potential(self):
        """
        This is the repulsive regime of the Lenard-Jones potential connected to a uniform potential.
        """
        cutoff = self.nucleosome_diameter * 2. ** (1. / 6.)  # Minimum of the LJ potential.
        energy = (
            'f * step(d * 2^(1/6) - r);'
            'f = 4.0 * ((d/r)^12 - (d/r)^6) + 1;'
            'd = radius1 + radius2'
        )
        force = openmm.CustomNonbondedForce(energy)
        force.addPerParticleParameter('radius')
        force.setCutoffDistance(cutoff)
        for i in range(self.n_total):
            force.addParticle([self.nucleosome_diameter / 2])
        for i in range(self.n_dna * (self.n_total - 1)):
            force.addParticle([self.dna_diameter / 2])

        self.add_exclusions(force)
        force.setNonbondedMethod(force.CutoffNonPeriodic)
        self.system.addForce(force)

    def add_terminal_potential(self):
        """
        Add a hard-core potential between the terminal nucleosomes and the rest of the chain.

        The potential is not added for the DNA beads to speed up the simulation. This is not a problem because
        the terminal bead is generally so large that it will overlap with the nucleosome beads first anyway.
        """
        # Implementing this as a bond is much faster than as a nonbonded force...
        force = openmm.CustomBondForce('f * step(d * 2^(1/6) - r);'
                                       'f = 4.0 * ((d/r)^12 - (d/r)^6) + 1;')
        force.addPerBondParameter('d')
        for i in [0, self.n_total - 1]:
            exclude = math.ceil(self.terminal_nucleosome_diameter / 2 / self.bond_length)
            for j in range(1+exclude, self.n_total-1-exclude):
                force.addBond(i, j, [(self.terminal_nucleosome_diameter + self.nucleosome_diameter) / 2])
        force.addBond(0, self.n_total - 1, [self.terminal_nucleosome_diameter])
        force.setUsesPeriodicBoundaryConditions(False)
        self.system.addForce(force)

    def add_tanh_potential(self, alpha, scale, r_c, cutoff_value=0.001, switch_value=0.01):
        """
        Add a tanh potential to the system with weights specified by `alpha`.
        """
        energy = (
            f"alpha(idx1, idx2) * f;"
            f"f = 0.5 * (1 + tanh({scale} * ({r_c} - r))) - {cutoff_value};"
        )
        force = openmm.CustomNonbondedForce(energy)
        force.addPerParticleParameter('idx')

        alpha = triu_to_full(alpha, k=2, fill=0.0)
        alpha = np.hstack([alpha, np.zeros((self.n, 1))])  # Add zeros for the cap particles.
        alpha = np.vstack([alpha, np.zeros(self.n+1)])
        alpha[-1, -1] = self.cap_cap
        alpha[-1, :-1] = self.cap_nuc
        alpha[:-1, -1] = self.cap_nuc
        alpha = alpha.flatten()
        alpha_tfunc = openmm.Discrete2DFunction(self.n + 1, self.n + 1, alpha)
        force.addTabulatedFunction('alpha', alpha_tfunc)

        for i in range(self.n_cap):
            force.addParticle([self.n])
        for i in range(self.n):
            force.addParticle([i])
        for i in range(self.n_cap):
            force.addParticle([self.n])
        for i in range(self.n_dna*(self.n_total - 1)):
            force.addParticle([1e42])  # Virtual sites also need to be added, but they are not actually used.

        self._add_nuc_nuc_tanh(force, scale, r_c, cutoff_value, switch_value)

    def add_uniform_tanh_potential(self, cutoff_value=0.001, switch_value=0.01):
        """
        Add a tanh potential to all nucleosome pairs with weight specified by `alpha`.
        """
        energy = (
            f"alpha(type1, type2) * f;"
            f"f = 0.5 * (1 + tanh({self.nuc_nuc_scale} * ({self.nuc_nuc_rc} - r))) - {cutoff_value};"
        )
        force = openmm.CustomNonbondedForce(energy)
        force.addPerParticleParameter('type')

        alpha = np.array([[self.nuc_nuc_attraction, self.cap_nuc_attraction],
                          [self.cap_nuc_attraction, self.cap_cap_attraction]])
        alpha = alpha.flatten()
        alpha_tfunc = openmm.Discrete2DFunction(2, 2, alpha)
        force.addTabulatedFunction('alpha', alpha_tfunc)

        for i in range(self.n_cap):
            force.addParticle([1])
        for i in range(self.n):
            force.addParticle([0])
        for i in range(self.n_cap):
            force.addParticle([1])
        for i in range(self.n_dna * (self.n_total - 1)):
            force.addParticle([1e42])  # Virtual sites also need to be added, but they are not actually used.

        self._add_nuc_nuc_tanh(force, self.nuc_nuc_scale, self.nuc_nuc_rc, cutoff_value, switch_value)

    def _add_nuc_nuc_tanh(self, force, scale, r_c, cutoff_value, switch_value):
        force.addInteractionGroup(range(self.n_total), range(self.n_total))
        self.add_exclusions(force)
        force.setNonbondedMethod(force.CutoffNonPeriodic)
        force.setUseSwitchingFunction(True)
        force.setSwitchingDistance(r_c - 1 / scale * np.arctanh(2 * switch_value - 1))
        force.setCutoffDistance(r_c - 1 / scale * np.arctanh(2 * cutoff_value - 1))
        self.system.addForce(force)
