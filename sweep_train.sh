#!/bin/bash

#SBATCH --account=ucb821_asc1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --time=11:59:00
#SBATCH --output=/scratch/alpine/%u/job_output/job_%a-%j.out
#SBATCH --mail-type=ALL
#SBATCH --partition=acpu
#SBATCH --qos=cpu-normal
#SBATCH --mem=28G
#SBATCH --threads-per-core=1

module purge

echo "Loading modules"
module load python/3.10.2
module load gcc

echo "Activating virtual environment"
source /projects/$USER/test/.venv/bin/activate
export TMPDIR=/scratch/alpine/$USER/temp_dir

echo "Running training script"

python3 /projects/yuna1623/MovingTargets/code/simple_structure_bounded/hyperparameter_search/train_sat_main_PBT.py \
    --n-cases 120 \
    --max-concurrent 30 \
    --launch-delay-s 3

echo "== End of Job =="
