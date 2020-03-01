#!/bin/env bash
#SBATCH -N 1      # nodes requested
#SBATCH -n 1      # tasks requested
#SBATCH -c 12      # cores requested
#SBATCH --mem=100000  # memory in Mb
#SBATCH -t 32:00:00  # time requested in hour:minute:second
#SBATCH --partition=gpu # request gpu partition specfically

srun -N 1 python /home/jiwu/interpretable-fundus/fundus_experiments/scripts/training_experiment.py with 'loss_setting = {"type": "focal_loss", "gamma": 1}'
srun -N 1 python /home/jiwu/interpretable-fundus/fundus_experiments/scripts/training_experiment.py with 'loss_setting = {"type": "focal_loss", "gamma": 2}'
srun -N 1 python /home/jiwu/interpretable-fundus/fundus_experiments/scripts/training_experiment.py with 'loss_setting = {"type": "focal_loss", "gamma": 3}'
srun -N 1 python /home/jiwu/interpretable-fundus/fundus_experiments/scripts/training_experiment.py with 'loss_setting = {"type": "focal_loss", "gamma": 4}' 
#unset DISPLAY XAUTHORITY 
#echo "Running ${@}"
#"${@}"
