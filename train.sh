CONFIG_FILE=/home/mawanda/projects/HuBMAP/configs/bisenetv2_fcn_fp16_4x4_1024x1024_160k_cityscapes.py
WORK_DIR=/home/mawanda/Documents/HuBMAP/experiments/only_mosaic

cd mmsegmentation
./tools/dist_train.sh ${CONFIG_FILE} 1 --work-dir ${WORK_DIR}