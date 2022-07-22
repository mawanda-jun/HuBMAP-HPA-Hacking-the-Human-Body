CONFIG_FILE=/home/mawanda/projects/HuBMAP/configs/segformer_mit-b2_512x512_160k_ade20k.py
WORK_DIR=/home/mawanda/Documents/HuBMAP/experiments/segformer_lovasz_geom_multires_morecolor

cd mmsegmentation
./tools/dist_train.sh ${CONFIG_FILE} 1 --work-dir ${WORK_DIR} --auto-resume
