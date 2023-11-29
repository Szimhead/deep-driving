#!/bin/sh

#BSUB -J test_ngpu
#BSUB -n 4
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=4GB]"
#BUSB -M 10GB
#BSUB -W 20:00
#BSUB -u s232449@dtu.dk
#BSUB -B
#BSUB -N
#BSUB -o out_test.out
#BSUB -e out_test.err
##BSUB -gpu "num=1:mode=exclusive_process"

export REPO=~/Deep-driving

if [[ ! -d ${REPO}/job_out ]]; then
   mkdir ${REPO}/job_out
fi

date=$(date +%Y%m%d_%H%M)
mkdir ${REPO}/runs/train/${date}
source ${REPO}/.venv/bin/activate

module load python3/3.10.12

module load cuda/11.8
module load cudnn/v8.6.0.163-prod-cuda-11.X
module load tensorrt/8.6.1.6-cuda-11.X
module load pandas/2.0.2-python-3.10.12
module load matplotlib
module load scipy/1.10.1-python-3.10.12

python3 -m pip install jproperties
python3 -m pip install Pillow
python3 -m pip install torch
python3 -m pip install tensorflow==2.12.0
python3 -m pip install scikit-learn
python3 -m pip install opencv-python
python3 -m pip install tqdm
# run training
python3 test.py
