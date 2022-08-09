import concurrent.futures
import os
from itertools import repeat
from pathlib import Path
from typing import Dict

import cv2
import pandas as pd
from PIL import Image
from tqdm import tqdm

from cropper import get_windows
from rle_scripts import rle_decode
from utils.shared_functions import load_validation_images
from utils.generic_functions import load_yaml
import argparse


def cut_and_save(
        img_path,
        rle_mask,
        scale_factor,
        crop_configurations,
        for_val,
        out_train_img,
        out_train_ann,
        out_val_img,
        out_val_ann
):
    img = cv2.imread(img_path)
    mask = rle_decode(rle_mask, (img.shape[1], img.shape[0]))
    # Resize image to given shape
    if scale_factor is not None:
        img_shape = img.shape
        dest_shape = [int(round(img_shape[0]*scale_factor)), int(round(img_shape[1]*scale_factor))]
        # Resize to 1536 so there are at maximum 9 tiles per image when dealing with 512x512
        dest_shape = [min(1536, dest_shape[0]), min(1536, dest_shape[1])]

        img = cv2.resize(img, dest_shape, interpolation=cv2.INTER_LANCZOS4)
        mask = cv2.resize(mask, dest_shape, interpolation=cv2.INTER_LANCZOS4)

    for crop_configuration in crop_configurations:
        img_windows = get_windows(img.shape[0], img.shape[1], *crop_configuration)
        # Augment total number of images when dealing with little ones
        # 2 is picked since the smallest class is already over represented
        if len(img_windows) < 2:
            img_windows *= 2
        for i, window in enumerate(img_windows):
            cropped_mask = mask[
                           window["row_off"]:window["row_off"] + window['height'],
                           window["col_off"]:window["col_off"] + window['width']]
            if cropped_mask.sum() > 200:  # There is at least a mask of 25x25 in this window!
                cropped_img = img[
                              window["row_off"]:window["row_off"] + window['height'],
                              window["col_off"]:window["col_off"] + window['width'],
                              ...]

                # Save images
                crop_configuration = [str(a) for a in crop_configuration]
                img_name = int(os.path.basename(img_path).split(".")[0])
                if img_name in for_val:
                    output_img_path = os.path.join(out_val_img,
                                                   str(img_name) + f"__{'_'.join(crop_configuration)}__{i}.png")
                    output_ann_path = os.path.join(out_val_ann,
                                                   str(img_name) + f"__{'_'.join(crop_configuration)}__{i}.png")
                else:
                    output_img_path = os.path.join(out_train_img,
                                                   str(img_name) + f"__{'_'.join(crop_configuration)}__{i}.png")
                    output_ann_path = os.path.join(out_train_ann,
                                                   str(img_name) + f"__{'_'.join(crop_configuration)}__{i}.png")

                cv2.imwrite(output_img_path, cropped_img)
                Image.fromarray(cropped_mask).save(output_ann_path, bits=1, optimize=True)
                # cv2.imwrite(output_ann_path, cropped_mask)


def run(base_path: Path, config_path: Path, resize: bool):
    config: Dict = load_yaml(config_path)
    crop_configurations = config["crop_configurations"]
    test_pixel_size = config["test_pixel_size"]

    # Keep some images for validation. I've selected those by looking
    validation_images = load_validation_images(config_path)

    if resize:
        if len(crop_configurations) > 1:
            output_dir = base_path / f'for_mmdetection_resized_multires_{crop_configurations[0][0]}'

        else:
            output_dir = base_path / f'for_mmdetection_resized_{crop_configurations[0][0]}'
    else:
        if len(crop_configurations) > 1:
            output_dir = base_path / f'for_mmdetection_multires_{crop_configurations[0][0]}'

        else:
            output_dir = base_path / f'for_mmdetection_{crop_configurations[0][0]}'
    ann_path = base_path / "train.csv"

    whole_annotations = pd.read_csv(ann_path)

    # img, mask = load_img_mask(train_annotations, base_path, train_annotations["id"][350])
    # plt.imshow(img)
    # plt.imshow(mask, cmap='coolwarm', alpha=0.5)

    # Save png format for fiftyone visualization
    out_train_img = output_dir / "img_dir" / "train" / "organ"
    out_val_img = output_dir / "img_dir" / "val" / "organ"
    out_train_ann = output_dir / "ann_dir" / "train" / "organ"
    out_val_ann = output_dir / "ann_dir" / "val" / "organ"

    os.makedirs(out_train_img, exist_ok=True)
    os.makedirs(out_val_img, exist_ok=True)
    os.makedirs(out_train_ann, exist_ok=True)
    os.makedirs(out_val_ann, exist_ok=True)

    img_paths = [str(base_path / "train_images" / f"{str(img_name)}.tiff") for img_name in
                 whole_annotations["id"]]
    rle_masks = [whole_annotations[whole_annotations["id"] == img_name]["rle"].iloc[-1] for img_name in
                 whole_annotations["id"]]
    categories = [whole_annotations[whole_annotations["id"] == img_name]["organ"].iloc[-1] for img_name in
                 whole_annotations["id"]]
    if resize:
        pixel_sizes = [whole_annotations[whole_annotations["id"] == img_name]["pixel_size"].iloc[-1] for img_name in
                 whole_annotations["id"]]
        scale_factors = [pixel_size/test_pixel_size[category] for pixel_size, category in zip(pixel_sizes, categories)]
    else:
        scale_factors = [None for _ in whole_annotations["id"]]
    with concurrent.futures.ProcessPoolExecutor(max_workers=os.cpu_count()) as worker:
        _ = list(tqdm(
            worker.map(
                cut_and_save,
                img_paths,
                rle_masks,
                scale_factors,
                repeat(crop_configurations),
                repeat(validation_images),
                repeat(out_train_img),
                repeat(out_train_ann),
                repeat(out_val_img),
                repeat(out_val_ann)
            ), total=len(img_paths), desc="Hubmap -> ADE20K"))


def main():
    parser = argparse.ArgumentParser(description="Dataset path parser")
    parser.add_argument('--path', action='store', type=str)
    parser.add_argument('--resize', action='store_true')
    args = parser.parse_args()
    # "/home/mawanda/Documents/HuBMAP/"
    base_path: Path = Path(args.path)
    resize: bool = args.resize
    config_path: Path = Path("dataset_tools/ade20k_config.yml")
    run(base_path, config_path, resize)


if "__main__" in __name__:
    main()
