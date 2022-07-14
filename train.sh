CONFIG_FILE=/home/mawanda/projects/HuBMAP/configs/upernet_r50_512x512_80k_ade20k.py
WORK_DIR=/home/mawanda/Documents/HuBMAP/experiments/upernet_baseline

cd mmsegmentation
./tools/dist_train.sh ${CONFIG_FILE} 1 --work-dir ${WORK_DIR}