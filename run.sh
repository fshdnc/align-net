#!/bin/bash
#SBATCH -J train
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=10:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:v100:1
#SBATCH --ntasks-per-node=1
#SBATCH --account=project_2002820
#SBATCH --output=/scratch/project_2002820/lihsin/align-lang/log-%j.out
#SBATCH --error=/scratch/project_2002820/lihsin/align-lang/log-%j.err

echo "START $SLURM_JOBID: $(date)"

function on_exit {
seff $SLURM_JOBID
gpuseff $SLURM_JOBID
echo "END $SLURM_JOBID: $(date)"
}
trap on_exit EXIT


module purge
module load pytorch/1.3.1
python3 train.py \
    --train-batch-size 64 \
    --learning-rate 1e-5

