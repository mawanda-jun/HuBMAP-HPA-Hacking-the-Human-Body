import os
import pandas as pd
from PIL import Image
from tqdm import tqdm
import cv2
from rle_scripts import rle_decode
import concurrent.futures as futures

def main():
    
    # Keep some images for validation. I've selected those by looking 
    kidney = [62, 164, 5099, 5102, 17422, 18426, 18777, 27298, 27468, 30876]
    prostate = [435, 4658, 4944, 7902, 8227, 14396, 22995, 29307, 32412]
    largeintestine = [10651, 11890, 12471, 25298, 25689, 31799]
    spleen = [10703, 12026, 19377, 19997, 32231]
    lung = [1878, 4301, 5086, 16564, 23252]

    for_val = [*kidney, *prostate, *largeintestine, *spleen, *lung]

    base_path = "/home/mawanda/Documents/HuBMAP/"
    output_dir = os.path.join(base_path, 'for_mmdetection')
    ann_path = os.path.join(base_path, "train.csv")
    
    whole_annotations = pd.read_csv(ann_path)

    # img, mask = load_img_mask(train_annotations, base_path, train_annotations["id"][350])
    # plt.imshow(img)
    # plt.imshow(mask, cmap='coolwarm', alpha=0.5)

    # Save png format for fiftyone visualization
    out_train_img = os.path.join(output_dir, "img_dir", "train", "organ")
    out_val_img = os.path.join(output_dir, "img_dir", "val", "organ")
    out_train_ann = os.path.join(output_dir, "ann_dir", "train", "organ")
    out_val_ann = os.path.join(output_dir, "ann_dir", "val", "organ")

    os.makedirs(out_train_img, exist_ok=True)
    os.makedirs(out_val_img, exist_ok=True)
    os.makedirs(out_train_ann, exist_ok=True)
    os.makedirs(out_val_ann, exist_ok=True)

    for img_name in tqdm(whole_annotations["id"], desc="Processing annotations...", total=len(whole_annotations["id"])):
        img = cv2.imread(os.path.join(base_path, "train_images", str(img_name) + ".tiff"))
        rle_mask = whole_annotations[whole_annotations["id"]==img_name]["rle"].iloc[-1]
        mask = rle_decode(rle_mask, (img.shape[1], img.shape[0])).T
        if img_name in for_val:
            output_img_path = os.path.join(out_val_img, str(img_name) + ".png")
            output_ann_path = os.path.join(out_val_ann, str(img_name) + ".png")
        else:
            output_img_path = os.path.join(out_train_img, str(img_name) + ".png")
            output_ann_path = os.path.join(out_train_ann, str(img_name) + ".png")
        cv2.imwrite(output_img_path, img)
        Image.fromarray(mask).save(output_ann_path)
        


if "__main__" in __name__:
    main()
