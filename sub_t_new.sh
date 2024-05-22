#!/bin/bash

#SBATCH -o Logs/%j.log
#SBATCH --time=1440
#SBATCH --ntasks=1
##SBATCH --partition=project79-a100-v2
#SBATCH --partition=shared-a100-v2,shared-a100
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=32

set -uxeC

# singularity container file name
EXAMPLE_CONTAINER_NAME="nvcr.io-nvidia-pytorch.23.04-py3.sif"

# example resources /sample/pytorch
CONTAINER_SRC_PATH="/sample/container/NGC/pytorch/$EXAMPLE_CONTAINER_NAME"

set +x
set -uC
source /etc/profile.d/modules.sh
module load singularity/3.5.3
module load openmpi/3.1.6

echo "start learning"
singularity exec --bind /group/project79:/group/project79 \
    --nv "$CONTAINER_SRC_PATH" \
    python main.py  --mode train --config configs/new_kitti.yaml
    
    #python openstereo/main.py --config ./configs/sttr/STTR_mid.yaml --scope val --restore_hint ./modelzoo/sceneflow/STTR-Stereo_FlyingThings3DSubset_epoch_08.pt # STTR
    #python openstereo/main.py --config ./configs/casnet/CasNet_mid.yaml --scope val --restore_hint ./modelzoo/sceneflow/CasNet_SceneFlow.pt # Casnet
    #python openstereo/main.py --config ./configs/psmnet/PSMNet_mid.yaml --scope val --restore_hint ./modelzoo/sceneflow/PSMNet_SceneFlow.pt #PSM
    #python openstereo/main.py --config ./configs/casnet/CasNet_sceneflow.yaml --scope val --restore_hint ./modelzoo/sceneflow/CasNet_SceneFlow.pt # Casnet
    #python openstereo/main.py --config ./configs/coex/CoExNet_mid.yaml --scope val --restore_hint ./modelzoo/sceneflow/CoExNet_SceneFlow_epoch_015.pt # COEX
    #python openstereo/main.py --config ./configs/coex/CoExNet_sceneflow.yaml --scope val --restore_hint ./modelzoo/sceneflow/CoExNet_SceneFlow_epoch_015.pt # COEX
    #python openstereo/main.py --config ./configs/raft/RAFT_sceneflow.yaml --scope val --restore_hint ./modelzoo/sceneflow/RAFT-Stereo_SceneFlow.pt # STTR
    
    #python openstereo/main.py --config ./configs/igev/igev_sceneflow.yaml --scope val --restore_hint ./modelzoo/sceneflow/IGEV_Sceneflow_epoch_090.pt
    #python openstereo/main.py --config ./configs/igev/igev_mid.yaml --scope val --restore_hint ./modelzoo/sceneflow/IGEV_Sceneflow_epoch_090.pt