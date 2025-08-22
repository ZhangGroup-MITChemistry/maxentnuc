#!/bin/bash
#SBATCH --time=72:00:00
#SBATCH --job-name remd
#SBATCH -o %j.log
#SBATCH -N 1
#SBATCH --ntasks-per-node=2
#SBATCH --gres=gpu:volta:2
#SBATCH --cpus-per-task=20

# There are 2 GPUs (and 40 cpus) per node on supercloud.
# You should generally request a whole node (gres=gpu:volta:2) and have 1 task per gpu (ntasks-per-node=2).
# You can then request 1-4 nodes depending on the number of parallel simulations you want to run.

# Load modules
module load mpi/openmpi-4.1.5
module load cuda/11.8
module load nccl/2.18.1-cuda11.8

export WORLD_SIZE=$((${SLURM_NNODES}*${SLURM_NTASKS_PER_NODE}))
echo "WORLD_SIZE=${WORLD_SIZE}"

# These flags tell MPI how to set up communication
export MPI_FLAGS="--tag-output --bind-to socket -map-by core -x PSM2_GPUDIRECT=1 -x NCCL_NET_GDR_LEVEL=5 -x NCCL_P2P_LEVEL=5 -x NCCL_NET_GDR_READ=1"

# Set some environment variables needed by torch.distributed
export MASTER_ADDR=$(hostname -s)
# Get unused port
export MASTER_PORT=$(python -c 'import socket; s=socket.socket(); s.bind(("", 0)); print(s.getsockname()[1]); s.close()')

echo "MASTER_ADDR : ${MASTER_ADDR}"
echo "MASTER_PORT : ${MASTER_PORT}"

mpirun ${MPI_FLAGS} python -m maxentnuc.simulation.mei config.yaml
