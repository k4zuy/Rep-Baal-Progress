#!/bin/bash
                           
#SBATCH --no-requeue
#SBATCH --partition=alpha
#SBATCH --nodes=1                   
#SBATCH --gres=gpu:1
#SBATCH --mem=16000
#SBATCH --mincpus=1
#SBATCH --time=6-23:59:59                            
#SBATCH --job-name=mean_serhyi_cifar10
#SBATCH --mail-type=ALL
#SBATCH --mail-user=kevin.kirsten@mailbox.tu-dresden.de
#SBATCH --output=../Experiments/results/logs/output-%x.out

module --force purge                          				
module load modenv/hiera CUDA/11.7.0 GCCcore/11.3.0 Python/3.10.4

source lib.sh

create_or_reuse_environment $SLURM_JOB_NAME

cd /home/keki996e/AL4ML/ACTIVE/OwnExperiments/Rep-Baal-Progress/Augmented/Experiments

python all_vgg_augmented_cifar10.py --tag mean_serhyi_cifar10
