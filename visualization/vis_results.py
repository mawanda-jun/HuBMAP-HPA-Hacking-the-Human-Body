import numpy as np
from rle_scripts import rle_encode, rle_decode
from cropper import get_windows
from tqdm import tqdm

# mmsegmentation install confirmation
# this is for demo purposes to show mmsegmentation working
from mmseg.apis import inference_segmentor, init_segmentor
import mmcv
import pandas as pd
from PIL import Image

import os

config_file = "/home/mawanda/Documents/HuBMAP/experiments/progressive_resized/1_inverted_segformer_RandomResizedCrop/1_dataset_RandomResizedCrop.py"
checkpoint_file = "/home/mawanda/Documents/HuBMAP/experiments/progressive_resized/1_inverted_segformer_RandomResizedCrop/iter_4182.pth"

# build the model from a config file and a checkpoint file
model = init_segmentor(config_file, checkpoint_file, device='cuda:0')

imgs = [
    "/home/mawanda/Documents/HuBMAP/for_mmdetection_resized_6000_inverted/img_dir/val/organ/5086__6000_6000_6000_6000__0.png"
]
anns = [
    "/home/mawanda/Documents/HuBMAP/for_mmdetection_resized_6000_inverted/ann_dir/val/organ/5086__6000_6000_6000_6000__0.png"
]

names, preds = [], []
for img_path, ori_mask_path in zip(imgs, anns):
    img = np.asarray(Image.open(img_path))
    if "inverted" in img_path:
        img = 255 - img
    ori_mask = np.asarray(Image.open(ori_mask_path))

    mask = np.zeros((img.shape[0], img.shape[1]))
    counter = np.zeros_like(mask)

    for window in tqdm(get_windows(img.shape[0], img.shape[1], 512, 512, 384, 384)):
        result = inference_segmentor(model, img[
            window['row_off']:window['row_off']+window['height'],
            window['col_off']:window['col_off']+window['width'],
            ...
        ])
        part_mask = result[0]
        mask[
            window['row_off']:window['row_off']+window['height'],
            window['col_off']:window['col_off']+window['width']
        ] += part_mask
        counter[
            window['row_off']:window['row_off']+window['height'],
            window['col_off']:window['col_off']+window['width']
        ] += 1
    
    good_indexes = mask/counter > 0.75
    mask = np.zeros_like(mask)
    mask[good_indexes] = 1
    # mask = mask.clip(0, 1)

    assert np.array_equal(mask, rle_decode(
        rle_encode(mask), shape=(mask.shape[0], mask.shape[1])))

    model.show_result(img_path, [ori_mask],
                      out_file="original.jpg", opacity=0.5)
    model.show_result(img_path, [mask],
                      out_file="inferred.jpg", opacity=0.5)
    # model.show_result("/home/mawanda/Documents/HuBMAP/for_mmdetection/img_dir/val/organ/8227.png", [mask], out_file="inferred_stained_from_PAS.jpg", opacity=0.5)
