import os
import pandas as pd
import tifffile as tiff
from tqdm import tqdm
import matplotlib.pyplot as plt
from datetime import date
from pycocotools import mask as mask_util
import cv2
import json
import numpy as np
from rle_scripts import rle_decode, rle_encode

organs_to_id = {
    'kidney': 0,
    "prostate": 1,
    "largeintestine": 2,
    "spleen": 3,
    "lung": 4
}
id_to_organ = {v: k for k, v in organs_to_id.items()}

def save_coco(output_filepath, images, annotations):
    json.dump({
        "info": {
        "description": "HubMAP dataset in COCO format",
        "url": "http://xviewdataset.org/",
        "version": "1.0",
        "year": "2022",
        "contributor": "Giovanni Cavallin",
        "date_created": str(date.today())
    },
        "licenses": [{
        "url": "https://creativecommons.org/licenses/by-nc-sa/4.0/",
        "id": 0,
        "name": "Attribution-NonCommercial-ShareAlike 4.0 International (CC BY-NC-SA 4.0)"
    }] ,
        "categories": [{
        "supercategory": "none",
        "id": 0,
        "name": "general_organ"
    }] ,
        # "categories": [{
        #     "supercategory": "none",
        #     "id": v,
        #     "name": k
        #     } for k, v in organs_to_id.items()], 
        "images": images,
        "annotations": annotations
    }, open(output_filepath, 'w'))


def load_img_mask(annotations, dataset_path, datapoint):
    input_image = cv2.imread(os.path.join(dataset_path, "train_images", str(datapoint) + ".tiff"))
    input_mask = rle_decode(annotations[annotations["id"]==datapoint]["rle"].iloc[-1], (input_image.shape[1], input_image.shape[0]))
    
    return input_image, input_mask

def main():
    base_path = "/home/mawanda/Documents/HuBMAP/"
    ann_path = os.path.join(base_path, "train.csv")
    
    whole_annotations = pd.read_csv(ann_path)

    # img, mask = load_img_mask(train_annotations, base_path, train_annotations["id"][350])
    # plt.imshow(img)
    # plt.imshow(mask, cmap='coolwarm', alpha=0.5)

    # Save png format for fiftyone visualization
    out_img_dir = os.path.join(base_path, "train_images_png")
    os.makedirs(out_img_dir, exist_ok=True)
    images = []
    annotations = []
    for i, img_name in tqdm(enumerate(whole_annotations["id"]), desc="Processing annotations...", total=len(whole_annotations["id"])):
        img = cv2.imread(os.path.join(base_path, "train_images", str(img_name) + ".tiff"))
        out_img_path = os.path.join(out_img_dir, str(img_name) + ".png")
        if not os.path.isfile(out_img_path):
            cv2.imwrite(out_img_path, img)
        rle_mask = whole_annotations[whole_annotations["id"]==img_name]["rle"].iloc[-1]
        organ = whole_annotations[whole_annotations["id"]==img_name]["organ"].iloc[-1]
        try:
            mask = rle_decode(rle_mask, (img.shape[1], img.shape[0]))
            c_rle = mask_util.encode(np.asfortranarray(mask))  # Encode mask back to rle - but now we have more information about mask thanks to pycocotools
        except Exception as e:
            print(img_name)
            raise e
        c_rle['counts'] = c_rle['counts'].decode('utf-8')  # Convert from binary to utf-8 for json memorizing
        area = mask_util.area(c_rle).item()  # calculate area
        bbox = mask_util.toBbox(c_rle).astype(int).tolist()  # calculate bbox
        # c_rle['counts'] = rle_mask

        annotation = {
            'segmentation': c_rle,
            'bbox': bbox,
            'area': area,
            'image_id': img_name,
            'category_id': 0,
            # 'category_id': organs_to_id[organ],
            'iscrowd': 1,
            'id': i,
        }

        img_dict = {
            "license": 0,
            "file_name": str(img_name) + ".png",
            "height": img.shape[1],
            "width": img.shape[0],
            "id": img_name
        }

        images.append(img_dict)
        annotations.append(annotation)
    
    output_filepath = os.path.join(base_path, "train_coco.json")
    val_output_filepath = os.path.join(base_path, "val_coco.json")
    print("Now saving...")
    
    # Keep some images for validation. I've selected those by looking 
    kidney = [62, 164, 5099, 5102, 17422, 18426, 18777, 27298, 27468, 30876]
    prostate = [435, 4658, 4944, 7902, 8227, 14396, 22995, 29307, 32412]
    largeintestine = [10651, 11890, 12471, 25298, 25689, 31799]
    spleen = [10703, 12026, 19377, 19997, 32231]
    lung = [1878, 4301, 5086, 16564, 23252]
    for_val = [*kidney, *prostate, *largeintestine, *spleen, *lung]
    train_images = []
    val_images = []
    train_annotations = []
    val_annotations = []
    val_found = 0
    for i, annotation in enumerate(annotations):
        if annotation["image_id"] in for_val:
            val_annotations.append(annotation)
            val_images.append(images[i])
            val_found += 1
        else:
            train_annotations.append(annotation)
            train_images.append(images[i])

    assert val_found == len(for_val), "Some val images has been written wrong..."
    save_coco(output_filepath, train_images, train_annotations)
    save_coco(val_output_filepath, val_images, val_annotations)
    
    print("File saved!")



if "__main__" in __name__:
    main()
