import os
import pandas as pd
from PIL import Image
from tqdm import tqdm
import cv2
from rle_scripts import rle_decode
from cropper import windows
from itertools import repeat
import concurrent.futures

def cut_and_save(
    img_path, 
    rle_mask,
    crop_configurations,
    for_val, 
    out_train_img, 
    out_train_ann, 
    out_val_img, 
    out_val_ann
):
    img = cv2.imread(img_path)
        
    mask = rle_decode(rle_mask, (img.shape[1], img.shape[0]))

    for crop_configuration in crop_configurations:
        img_windows = windows(img.shape[0], img.shape[1], *crop_configuration)
        for i, window in enumerate(img_windows):
            cropped_mask = mask[
                    window["row_off"]:window["row_off"]+window['height'],
                    window["col_off"]:window["col_off"]+window['width']]
            if cropped_mask.sum() > 625: # There is at least a mask of 25x25 in this window!
                cropped_img = img[
                    window["row_off"]:window["row_off"]+window['height'],
                    window["col_off"]:window["col_off"]+window['width'],
                ...]

                # Save images
                crop_configuration = [str(a) for a in crop_configuration]
                img_name = os.path.basename(img_path).split(".")[0]
                if img_name in for_val:
                    output_img_path = os.path.join(out_val_img, str(img_name) + f"__{'_'.join(crop_configuration)}__{i}.png")
                    output_ann_path = os.path.join(out_val_ann, str(img_name) + f"__{'_'.join(crop_configuration)}__{i}.png")
                else:
                    output_img_path = os.path.join(out_train_img, str(img_name) + f"__{'_'.join(crop_configuration)}__{i}.png")
                    output_ann_path = os.path.join(out_train_ann, str(img_name) + f"__{'_'.join(crop_configuration)}__{i}.png")
                
                cv2.imwrite(output_img_path, cropped_img)
                Image.fromarray(cropped_mask).save(output_ann_path, bits=1, optimize=True)
                # cv2.imwrite(output_ann_path, cropped_mask)


def main(base_path):
    crop_configurations = [
        [512, 512, 256, 256],
        [640, 640, 320, 320],
        [768, 768, 384, 384],
        [896, 896, 448, 448],
        [1024, 1024, 512, 512],
        [1024, 512, 512, 256],
        [512, 1024, 256, 512]
    ]
    
    # Keep some images for validation. I've selected those by looking 
    kidney = [62, 164, 5099, 5102, 17422, 18426, 18777, 27298, 27468, 30876]
    prostate = [435, 4658, 4944, 7902, 8227, 14396, 22995, 29307, 32412]
    largeintestine = [10651, 11890, 12471, 25298, 25689, 31799]
    spleen = [10703, 12026, 19377, 19997, 32231]
    lung = [1878, 4301, 5086, 16564, 23252]

    for_val = [*kidney, *prostate, *largeintestine, *spleen, *lung]

    if len(crop_configurations) > 1:
        output_dir = os.path.join(base_path, f'for_mmdetection_multires_{crop_configurations[0][0]}')
    else:
        output_dir = os.path.join(base_path, f'for_mmdetection_{crop_configurations[0][0]}')
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

    img_paths = [os.path.join(base_path, "train_images", str(img_name) + ".tiff") for img_name in whole_annotations["id"]]
    rle_masks = [whole_annotations[whole_annotations["id"]==img_name]["rle"].iloc[-1] for img_name in whole_annotations["id"]]
    with concurrent.futures.ProcessPoolExecutor(max_workers=os.cpu_count()) as worker:
        _ = list(tqdm(
            worker.map(
                cut_and_save,
                img_paths, 
                rle_masks,
                repeat(crop_configurations),
                repeat(for_val), 
                repeat(out_train_img), 
                repeat(out_train_ann), 
                repeat(out_val_img),
                repeat(out_val_ann)
        ), total=len(img_paths)))
            
            # for i, window in enumerate(windows(img.shape[0], img.shape[1], *crop_configuration)):
            #     window['num'] = i
            #     cut_and_save(
            #         img, 
            #         mask, 
            #         window, 
            #         img_name, 
            #         for_val, 
            #         out_train_img, 
            #         out_train_ann, 
            #         out_val_img,
            #         out_val_ann)


if "__main__" in __name__:
    base_path = "/home/mawanda/Documents/HuBMAP/"
    main(base_path)
