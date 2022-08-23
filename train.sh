# # echo Sleeping for 1h30min...
# # sleep 5400
cd mmsegmentation

# # ALL
# CONFIG_FILE=/home/mawanda/projects/HuBMAP/configs/resized_entire_b2_1024/14_dataset_all.py
# WORK_DIR=/home/mawanda/Documents/HuBMAP/experiments/resized_entire_b2_1024/14_inverted_segformer_all
# ./tools/dist_train.sh ${CONFIG_FILE} 1 --work-dir ${WORK_DIR}
# python ../test_model.py --path ${WORK_DIR}

# # NOAUG
# CONFIG_FILE=/home/mawanda/projects/HuBMAP/configs/resized_entire_b2_1024/0_dataset_NoAug.py
# WORK_DIR=/home/mawanda/Documents/HuBMAP/experiments/resized_entire_b2_1024/0_inverted_segformer_noaug
# ./tools/dist_train.sh ${CONFIG_FILE} 1 --work-dir ${WORK_DIR}
# python ../test_model.py --path ${WORK_DIR}

CONFIG_FILE=/home/mawanda/projects/HuBMAP/configs/resized_entire_b2_1024/1_dataset_RandomResizedCrop.py
WORK_DIR=/home/mawanda/Documents/HuBMAP/experiments/resized_entire_b2_1024_fastest/1_inverted_segformer_RandomResizedCrop
./tools/dist_train.sh ${CONFIG_FILE} 1 --work-dir ${WORK_DIR}
python ../test_model.py --path ${WORK_DIR}

# CONFIG_FILE=/home/mawanda/projects/HuBMAP/configs/resized_entire_b2_1024/2_dataset_RandomBrightness.py
# WORK_DIR=/home/mawanda/Documents/HuBMAP/experiments/resized_entire_b2_1024/2_inverted_segformer_RandomBrightness
# ./tools/dist_train.sh ${CONFIG_FILE} 1 --work-dir ${WORK_DIR}
# python ../test_model.py --path ${WORK_DIR}

# CONFIG_FILE=/home/mawanda/projects/HuBMAP/configs/resized_entire_b2_1024/3_dataset_RGBShift.py
# WORK_DIR=/home/mawanda/Documents/HuBMAP/experiments/resized_entire_b2_1024/3_inverted_segformer_RGBShift
# ./tools/dist_train.sh ${CONFIG_FILE} 1 --work-dir ${WORK_DIR}
# python ../test_model.py --path ${WORK_DIR}

# CONFIG_FILE=/home/mawanda/projects/HuBMAP/configs/resized_entire_b2_1024/4_dataset_HueSaturation.py
# WORK_DIR=/home/mawanda/Documents/HuBMAP/experiments/resized_entire_b2_1024/4_inverted_segformer_HueSaturation
# ./tools/dist_train.sh ${CONFIG_FILE} 1 --work-dir ${WORK_DIR}
# python ../test_model.py --path ${WORK_DIR}

# CONFIG_FILE=/home/mawanda/projects/HuBMAP/configs/resized_entire_b2_1024/5_dataset_RandomGamma.py
# WORK_DIR=/home/mawanda/Documents/HuBMAP/experiments/resized_entire_b2_1024/5_inverted_segformer_RandomGamma_longer
# ./tools/dist_train.sh ${CONFIG_FILE} 1 --work-dir ${WORK_DIR}
# python ../test_model.py --path ${WORK_DIR}


# CONFIG_FILE=/home/mawanda/projects/HuBMAP/configs/resized_entire_b2_1024/6_dataset_RandomCLAHE.py
# WORK_DIR=/home/mawanda/Documents/HuBMAP/experiments/resized_entire_b2_1024/6_inverted_segformer_RandomCLAHE
# ./tools/dist_train.sh ${CONFIG_FILE} 1 --work-dir ${WORK_DIR}
# python ../test_model.py --path ${WORK_DIR}

# CONFIG_FILE=/home/mawanda/projects/HuBMAP/configs/resized_entire_b2_1024/7_dataset_ImageCompression.py
# WORK_DIR=/home/mawanda/Documents/HuBMAP/experiments/resized_entire_b2_1024/7_inverted_segformer_ImageCompression
# ./tools/dist_train.sh ${CONFIG_FILE} 1 --work-dir ${WORK_DIR}
# python ../test_model.py --path ${WORK_DIR}

# CONFIG_FILE=/home/mawanda/projects/HuBMAP/configs/resized_entire_b2_1024/8_dataset_ChannelShuffle.py
# WORK_DIR=/home/mawanda/Documents/HuBMAP/experiments/resized_entire_b2_1024/8_inverted_segformer_ChannelShuffle
# ./tools/dist_train.sh ${CONFIG_FILE} 1 --work-dir ${WORK_DIR}
# python ../test_model.py --path ${WORK_DIR}

# CONFIG_FILE=/home/mawanda/projects/HuBMAP/configs/resized_entire_b2_1024/9_dataset_Blur.py
# WORK_DIR=/home/mawanda/Documents/HuBMAP/experiments/resized_entire_b2_1024/9_inverted_segformer_Blur
# ./tools/dist_train.sh ${CONFIG_FILE} 1 --work-dir ${WORK_DIR}
# python ../test_model.py --path ${WORK_DIR}

# CONFIG_FILE=/home/mawanda/projects/HuBMAP/configs/resized_entire_b2_1024/10_dataset_Elastic.py
# WORK_DIR=/home/mawanda/Documents/HuBMAP/experiments/resized_entire_b2_1024/10_inverted_segformer_Elastic
# ./tools/dist_train.sh ${CONFIG_FILE} 1 --work-dir ${WORK_DIR}
# python ../test_model.py --path ${WORK_DIR}

# CONFIG_FILE=/home/mawanda/projects/HuBMAP/configs/resized_entire_b2_1024/11_dataset_Grid.py
# WORK_DIR=/home/mawanda/Documents/HuBMAP/experiments/resized_entire_b2_1024/11_inverted_segformer_Grid
# ./tools/dist_train.sh ${CONFIG_FILE} 1 --work-dir ${WORK_DIR}
# python ../test_model.py --path ${WORK_DIR}

# CONFIG_FILE=/home/mawanda/projects/HuBMAP/configs/resized_entire_b2_1024/12_dataset_Optical.py
# WORK_DIR=/home/mawanda/Documents/HuBMAP/experiments/resized_entire_b2_1024/12_inverted_segformer_Optical
# ./tools/dist_train.sh ${CONFIG_FILE} 1 --work-dir ${WORK_DIR}
# python ../test_model.py --path ${WORK_DIR}

# CONFIG_FILE=/home/mawanda/projects/HuBMAP/configs/resized_entire_b2_1024/13_dataset_stain.py
# WORK_DIR=/home/mawanda/Documents/HuBMAP/experiments/resized_entire_b2_1024/13_inverted_segformer_stain
# ./tools/dist_train.sh ${CONFIG_FILE} 1 --work-dir ${WORK_DIR}
# python ../test_model.py --path ${WORK_DIR}