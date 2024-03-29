#!/bin/bash
#SBATCH -J train-con
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=72:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:v100:1
#SBATCH --ntasks-per-node=1
#SBATCH --account=project_2002820
#SBATCH --output=/scratch/project_2002820/lihsin/align-lang/log-con-%j.out
#SBATCH --error=/scratch/project_2002820/lihsin/align-lang/log-con-%j.err

echo "START $SLURM_JOBID: $(date)"

function on_exit {
seff $SLURM_JOBID
gpuseff $SLURM_JOBID
echo "END $SLURM_JOBID: $(date)"
}
trap on_exit EXIT

module purge
module load pytorch/1.3.1
python3 continue_train.py \
    --model-ckpt model_20210113-220042.pt \
    --skip-batch 12000 \
    --train-batch-size 32 \
    --learning-rate 1e-5

