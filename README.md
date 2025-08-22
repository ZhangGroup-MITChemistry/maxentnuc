# Maximum entropy inversion simulations of chromatin at nucleosome resolution

The core simulation code is in `maxentnuc/simulation`, simulation analysis modules are
in `maxentnuc/analysis`, most figure generation code is in `figures`, vmd scripts
to generate renderings are in `visualizations`, and a small demo is in `demo`.

See the README in `maxentnuc/simulation` for a description of the simulation methodology.

## Installation

Note that while small test systems can be run on a CPU, running meaningful systems is computationally demanding
and practically requires a GPU or ideally several GPUs.

This package depends on the neighbor-balance package, so you must first
install neighbor-balance and then this package.

```bash
conda create -n maxentnuc python=3.12 pip
conda activate maxentnuc

git clone https://github.com/ZhangGroup-MITChemistry/neighbor-balance
cd neighbor-balance
pip install .

git clone https://github.com/ZhangGroup-MITChemistry/maxentnuc
cd maxentnuc
pip install .
```

The software versions used in our paper can be seen in `environment.yml`.

## Running simulations

To run a simulation, you will need a cooler file containing the reference contact
data to be reproduced and a yaml formatted config file specifying the parameters
used in the simulation. See below for an example config file. You should only
need to adjust the first 3 parameters `contact_map_path`, `capture_probes_path`,
and `region`.

```yaml
contact_map_path: /home/joepaggi/share/rcmc/WT_minus_inward.mcool
capture_probes_path: /home/joepaggi/share/rcmc/captureprobes_mm39.bed
region: 'chr5:31,267,000-32,372,000'

max_iter: 35
error_tolerance: 0.00

normalization: 'mask'
normalization_params:
    normalization:
        'max_prob': 0.95
        'neighbor_prob': 0.5
        'bw': 1
    ice:
        'mad_max': 2
        'min_nnz': 100
        'correct_for_flanks': True
    neighbor: True
smoothing: coarsen

optimizer: 'gradient_descent'
optimizer_params:
    learning_rate: 0.3
    beta: 0.75
    initial_alpha: 0.0
    warmup_t: 5

model: polymer
model_params:
    bond_length: 21
    nucleosome_diameter: 11
    angle_constant: 1.0
    nuc_nuc_attraction: -1.0
    n_dna: 5
    n_cap: 50
    dna_diameter: 5
    terminal_nucleosome_diameter: 300
    nuc_nuc_scale: 0.381
    nuc_nuc_rc: 15
atom_selection: 'name NUC'

n_trajectories: 4
n_steps: 11_000_000
report_interval: 10_000
n_steps_burnin: 100
platform: CUDA
friction: 0.01
dt: 0.125
```

Given these input files, you can run the optimization using a submission script like the one below. However,
you will need to update some of the slurm parameters based on your computing environment. The below example
parallelizes the job across 4 processes. The total number of trajectories is equal to the number of processes
times `n_trajectories` from the config file. Therefore, to run an equivalent optimization with different numbers
of processes, you need to correpondingly update `n_trajectories`. For example, if you only have one GPU, then
set `-np 1` and `n_trajectories: 16`.

For a 1 MB system, the full optimization takes approximately 2 days using 4 l40s gpus.

```bash
#!/bin/bash
#SBATCH --job-name ppm1g
#SBATCH --dependency=SINGLETON
#SBATCH --time=48:00:00
#SBATCH --partition mit_preemptable
#SBATCH -o %j.log
#SBATCH -N 1
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:l40s:4
#SBATCH --cpus-per-task=10

mpirun -np 4 python -m maxentnuc.simulation.mei config.yaml
```

## Demo

A small demo that can be run relatively quickly on CPU is available in `demo/`.

```bash
cd demo
sh run_demo.sh
```

This runs an optimization for a 20 kb region. On a typical computer, it should take
about 10 minutes per iteration and reasonable convergence is achieved in ~5 iterations,
however close agreement is hard to achieve with short simulations.

When each iteration is completed, the code produces a file comparing the simulated
contact map to the reference map (e.g. 000_contact_map.png). You should see that
the simulated contact map approaches the reference map as the optimization progresses.

For this example, a precomputed reference contact map is provided, however, typically
one begins with a cooler file and the contact map is computed before beginning the
simulation.
