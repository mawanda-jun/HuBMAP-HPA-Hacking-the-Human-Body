import argparse
from pathlib import Path
import time
import numpy as np
from tqdm import tqdm
import cv2
from PIL import Image
import torch
from torch.utils.tensorboard import SummaryWriter

# mmsegmentation install confirmation
# this is for demo purposes to show mmsegmentation working
from mmseg.apis import inference_segmentor, init_segmentor
import os
from utils.generic_functions import load_yaml


def dice_coef(y_true, y_pred):
    y_true_f = y_true.flatten()
    y_pred_f = y_pred.flatten()
    intersection = np.sum(y_true_f * y_pred_f)
    smooth = 0.0001
    return (2. * intersection + smooth) / (np.sum(y_true_f) + np.sum(y_pred_f) + smooth)

def main(
    img_test_folder,
    ann_test_folder,
    config_file,
    checkpoint_file,
    cat_config
):
    categories = ["kidney", "prostate", "largeintestine", "spleen", "lung"]
    cat_config_reversed = {}
    for category in categories:
        for v in cat_config[category]:
            cat_config_reversed[v] = category

    per_category_mdice = {k: 0. for k in categories}

    # build the model from a config file and a checkpoint file
    model = init_segmentor(str(config_file), str(checkpoint_file), device='cuda')

    files = [file for file in os.listdir(img_test_folder)]
    files_categories = [cat_config_reversed[int(file.split("__")[0])] for file in files]
    images = [str(img_test_folder / file) for file in files]
    annotations = [str(ann_test_folder / file) for file in files]

    mdice = 0.
    for img_path, ori_mask_path, category in tqdm(zip(images, annotations, files_categories), desc=os.path.basename(checkpoint_file)):
        img = cv2.imread(img_path)
        if "inverted" in img_path:
            img = 255 - img
        ori_mask = np.asarray(Image.open(ori_mask_path))

        mask = np.zeros((img.shape[0], img.shape[1]))
        counter = np.zeros_like(mask)
        mask = inference_segmentor(model, img)[0]
        # for window in get_windows(img.shape[0], img.shape[1], 512, 512, 384, 384):
        #     result = inference_segmentor(model, img[
        #         window['row_off']:window['row_off']+window['height'],
        #         window['col_off']:window['col_off']+window['width'],
        #         ...
        #     ])
        #     part_mask = result[0]
        #     mask[
        #         window['row_off']:window['row_off']+window['height'],
        #         window['col_off']:window['col_off']+window['width']
        #     ] += part_mask
        #     counter[
        #         window['row_off']:window['row_off']+window['height'],
        #         window['col_off']:window['col_off']+window['width']
        #     ] += 1
        
        # good_indexes = mask/counter > 0.75
        # mask = np.zeros_like(mask)
        # mask[good_indexes] = 1
        
        # Calculate metric
        dc = dice_coef(ori_mask, mask)
        per_category_mdice[category] += dc
        mdice += dc
    
    per_category_mdice = {k:v/len(cat_config[k]) for k, v in per_category_mdice.items()}
    per_category_mdice['average'] = mdice / len(files)
    return per_category_mdice


if "__main__" in __name__:
    parser = argparse.ArgumentParser()
    parser.add_argument("--path")
    args = parser.parse_args()
    base_path = Path(args.path)
    # base_path = Path("/home/mawanda/Documents/HuBMAP/experiments/progressive_resized_entire/4_inverted_segformer_HueSaturation")
    config_file = base_path / [file for file in os.listdir(base_path) if ".py" in file][0]
    img_test_folder = Path("/home/mawanda/Documents/HuBMAP/for_mmdetection_resized_5000_inverted/img_dir/val/organ")
    ann_test_folder = Path("/home/mawanda/Documents/HuBMAP/for_mmdetection_resized_5000_inverted/ann_dir/val/organ")
    writer = SummaryWriter(log_dir=base_path / "tf_logs")

    # Configuration file for list of image categories
    cat_config = load_yaml("/home/mawanda/projects/HuBMAP/dataset_tools/ade20k_config.yml")

    # Find checkpoint files
    inferred_checkpoints = 0
    # while True:
    checkpoint_files = [file for file in os.listdir(str(base_path)) if "iter" in file]
    checkpoint_files.sort(key=lambda x: int(x.split("_")[1].split(".")[0]))
    # if len(checkpoint_files) > inferred_checkpoints:
        # print("Found new checkpoints! Now proceeding testing dataset...")
    for checkpoint_file in checkpoint_files[inferred_checkpoints:]:
        epoch_mdice = main(
            img_test_folder,
            ann_test_folder,
            config_file,
            base_path / checkpoint_file,
            cat_config
        )
        writer.add_scalars("test/mDice", epoch_mdice, int(checkpoint_file.split("_")[1].split(".")[0]))
        writer.flush()
        inferred_checkpoints += 1
    # else:
    #     for i in range(100, 0, -1):
    #         print(f"Waiting for new checkpoints. Sleeping for {i} s", end="\r", flush=True)
    #         time.sleep(1)
