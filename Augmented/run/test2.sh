#!/bin/bash
                           
#SBATCH --no-requeue
#SBATCH --partition=alpha
#SBATCH --nodes=1                   
#SBATCH --gres=gpu:1
#SBATCH --mem=16000
#SBATCH --mincpus=1
#SBATCH --time=6-23:59:59                            
#SBATCH --mail-type=ALL
#SBATCH --mail-user=kevin.kirsten@mailbox.tu-dresden.de
#SBATCH --output=output-r2_%x.out

module --force purge                          				
module load modenv/hiera CUDA/11.7.0 GCCcore/11.3.0 Python/3.10.4

python "${SLURM_JOB_NAME}.py" --tag "r2_${SLURM_JOB_NAME}"