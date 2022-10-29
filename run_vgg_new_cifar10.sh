#!/bin/bash

#SBATCH --no-requeue
#SBATCH --partition=alpha
#SBATCH --nodes=1                   
#SBATCH --gres=gpu:1
#SBATCH --mem=8000
#SBATCH --mincpus=1
#SBATCH --time=72:00:00                             
#SBATCH --job-name=vgg_new_cifar10
#SBATCH --mail-type=ALL
#SBATCH --mail-user=kevin.kirsten@mailbox.tu-dresden.de
#SBATCH --output=output-%x.out

module --force purge                          				
module load modenv/hiera CUDA/11.7.0 GCCcore/11.3.0 Python/3.10.4

if [ -d "/scratch/ws/1/keki996e-ws_run_vgg_new" ] 
then
    echo "Workspace exists and will be used"
    source /scratch/ws/1/keki996e-ws_run_vgg_new/pyenv/bin/activate
else
    WS_NAME="ws_run_vgg_new"
    FS_NAME="scratch"
    DURATION=30

    echo "Creating new environment $WS_NAME in FS $FS_NAME for $DURATION"
    WS_PATH=$(ws_allocate -F $FS_NAME $WS_NAME $DURATION)
    virtualenv $WS_PATH/pyenv
    source $WS_PATH/pyenv/bin/activate
    pip install --upgrade pip
    pip install torch torchvision tensorboard
    pip install baal tqdm Pillow
fi

python vgg_new_cifar10.py
