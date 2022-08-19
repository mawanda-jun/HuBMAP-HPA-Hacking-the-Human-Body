# # echo Sleeping for 1h30min...
# # sleep 5400
cd mmsegmentation

# ## NOAUG
# CONFIG_FILE=/home/mawanda/projects/HuBMAP/configs/progressive_resized_entire/0_dataset_NoAug.py
# WORK_DIR=/home/mawanda/Documents/HuBMAP/experiments/progressive_resized_entire/0_inverted_segformer_noaug

# ./tools/dist_train.sh ${CONFIG_FILE} 1 --work-dir ${WORK_DIR}python ../test_model.py --path ${WORK_DIR}


# CONFIG_FILE=/home/mawanda/projects/HuBMAP/configs/progressive_resized_entire/1_dataset_RandomResizedCrop.py
# WORK_DIR=/home/mawanda/Documents/HuBMAP/experiments/progressive_resized_entire/1_inverted_segformer_RandomResizedCrop
# ./tools/dist_train.sh ${CONFIG_FILE} 1 --work-dir ${WORK_DIR}
# python ../test_model.py --path ${WORK_DIR}

# CONFIG_FILE=/home/mawanda/projects/HuBMAP/configs/progressive_resized_entire/2_dataset_RandomBrightness.py
# WORK_DIR=/home/mawanda/Documents/HuBMAP/experiments/progressive_resized_entire/2_inverted_segformer_RandomBrightness
# ./tools/dist_train.sh ${CONFIG_FILE} 1 --work-dir ${WORK_DIR}
# python ../test_model.py --path ${WORK_DIR}

# CONFIG_FILE=/home/mawanda/projects/HuBMAP/configs/progressive_resized_entire/3_dataset_RGBShift.py
# WORK_DIR=/home/mawanda/Documents/HuBMAP/experiments/progressive_resized_entire/3_inverted_segformer_RGBShift
# ./tools/dist_train.sh ${CONFIG_FILE} 1 --work-dir ${WORK_DIR}
# python ../test_model.py --path ${WORK_DIR}

# CONFIG_FILE=/home/mawanda/projects/HuBMAP/configs/progressive_resized_entire/4_dataset_HueSaturation.py
# WORK_DIR=/home/mawanda/Documents/HuBMAP/experiments/progressive_resized_entire/4_inverted_segformer_HueSaturation
# ./tools/dist_train.sh ${CONFIG_FILE} 1 --work-dir ${WORK_DIR}
# python ../test_model.py --path ${WORK_DIR}

# CONFIG_FILE=/home/mawanda/projects/HuBMAP/configs/progressive_resized_entire/5_dataset_RandomGamma.py
# WORK_DIR=/home/mawanda/Documents/HuBMAP/experiments/progressive_resized_entire/5_inverted_segformer_RandomGamma_longer
# ./tools/dist_train.sh ${CONFIG_FILE} 1 --work-dir ${WORK_DIR}
# python ../test_model.py --path ${WORK_DIR}


CONFIG_FILE=/home/mawanda/projects/HuBMAP/configs/progressive_resized_entire/6_dataset_RandomCLAHE.py
WORK_DIR=/home/mawanda/Documents/HuBMAP/experiments/progressive_resized_entire/6_inverted_segformer_RandomCLAHE
./tools/dist_train.sh ${CONFIG_FILE} 1 --work-dir ${WORK_DIR}
python ../test_model.py --path ${WORK_DIR}

CONFIG_FILE=/home/mawanda/projects/HuBMAP/configs/progressive_resized_entire/7_dataset_ImageCompression.py
WORK_DIR=/home/mawanda/Documents/HuBMAP/experiments/progressive_resized_entire/7_inverted_segformer_ImageCompression
./tools/dist_train.sh ${CONFIG_FILE} 1 --work-dir ${WORK_DIR}
python ../test_model.py --path ${WORK_DIR}

CONFIG_FILE=/home/mawanda/projects/HuBMAP/configs/progressive_resized_entire/8_dataset_ChannelShuffle.py
WORK_DIR=/home/mawanda/Documents/HuBMAP/experiments/progressive_resized_entire/8_inverted_segformer_ChannelShuffle
./tools/dist_train.sh ${CONFIG_FILE} 1 --work-dir ${WORK_DIR}
python ../test_model.py --path ${WORK_DIR}

CONFIG_FILE=/home/mawanda/projects/HuBMAP/configs/progressive_resized_entire/9_dataset_Blur.py
WORK_DIR=/home/mawanda/Documents/HuBMAP/experiments/progressive_resized_entire/9_inverted_segformer_Blur
./tools/dist_train.sh ${CONFIG_FILE} 1 --work-dir ${WORK_DIR}
python ../test_model.py --path ${WORK_DIR}

CONFIG_FILE=/home/mawanda/projects/HuBMAP/configs/progressive_resized_entire/10_dataset_Elastic.py
WORK_DIR=/home/mawanda/Documents/HuBMAP/experiments/progressive_resized_entire/10_inverted_segformer_Elastic
./tools/dist_train.sh ${CONFIG_FILE} 1 --work-dir ${WORK_DIR}
python ../test_model.py --path ${WORK_DIR}

CONFIG_FILE=/home/mawanda/projects/HuBMAP/configs/progressive_resized_entire/11_dataset_Grid.py
WORK_DIR=/home/mawanda/Documents/HuBMAP/experiments/progressive_resized_entire/11_inverted_segformer_Grid
./tools/dist_train.sh ${CONFIG_FILE} 1 --work-dir ${WORK_DIR}
python ../test_model.py --path ${WORK_DIR}

CONFIG_FILE=/home/mawanda/projects/HuBMAP/configs/progressive_resized_entire/12_dataset_Optical.py
WORK_DIR=/home/mawanda/Documents/HuBMAP/experiments/progressive_resized_entire/12_inverted_segformer_Optical
./tools/dist_train.sh ${CONFIG_FILE} 1 --work-dir ${WORK_DIR}
python ../test_model.py --path ${WORK_DIR}

CONFIG_FILE=/home/mawanda/projects/HuBMAP/configs/progressive_resized_entire/13_dataset_stain.py
WORK_DIR=/home/mawanda/Documents/HuBMAP/experiments/progressive_resized_entire/13_inverted_segformer_stain
./tools/dist_train.sh ${CONFIG_FILE} 1 --work-dir ${WORK_DIR}
python ../test_model.py --path ${WORK_DIR}