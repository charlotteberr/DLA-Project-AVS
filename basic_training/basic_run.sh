#!/bin/bash

#SBATCH --account=ucb821_asc1
#SBATCH --job-name=dla-basic-5mintest
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=00:05:00
#SBATCH --output=/scratch/alpine/chbe6195/job_output/job_%j.out
#SBATCH --mail-type=ALL
#SBATCH --partition=acpu
#SBATCH --qos=cpu-testing
#SBATCH --mem=16G
#SBATCH --threads-per-core=1

module purge

echo "Loading modules"
module load python/3.10.2
module load gcc

echo "Activating virtual environment"
source /projects/chbe6195/.venv/bin/activate
export TMPDIR=/scratch/alpine/chbe6195/temp_dir

echo "Running training script"

python3 /projects/chbe6195/DLA-Project-AVS/basic_training/pbt_basic_imaging.py \
    --n-cases 120 \
    --max-concurrent 30 \
    --launch-delay-s 3

echo "== End of Job =="
