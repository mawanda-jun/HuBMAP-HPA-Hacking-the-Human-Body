import os
import concurrent.futures
from tqdm import tqdm
import staintools
from PIL import Image
import numpy as np


# Retrieve example images
# stainings_path = "/home/mawanda/Documents/HuBMAP/stainings"
# ref_images = [np.asarray(Image.open(os.path.join(stainings_path, file)).convert("RGB")) for file in os.listdir(stainings_path)]
ref_image = np.asarray(Image.open("/home/mawanda/Documents/HuBMAP/test_images/10078.tiff").convert("RGB"))

target = staintools.LuminosityStandardizer.standardize(ref_image) 

def stain_img(input_img):
    dirname = os.path.dirname(input_img)
    filename = os.path.basename(input_img)
    new_filename = filename.split(".")[0] + "_PAS.png"
    new_filepath = os.path.join(dirname, new_filename)
    if not os.path.isfile(new_filepath):
        try:
            image = np.asarray(Image.open(input_img))
            to_transform = staintools.LuminosityStandardizer.standardize(image.copy()) # altrimenti modifica image
            # Stain normalize
            normalizer = staintools.StainNormalizer(method='vahadane')
            normalizer.fit(target)
            output_img = normalizer.transform(to_transform)
            Image.fromarray(output_img).save(new_filepath)
        except staintools.miscellaneous.exceptions.TissueMaskException as e:
            print(f"Cannot find enough data in {input_img}, deleting image...")
            os.remove(input_img)
            ann_path = input_img.replace("img", "ann")
            os.remove(ann_path)

def prepare_folders(input_dir, output_dir):
    in_train_img = os.path.join(input_dir, "img_dir", "train", "organ")
    in_val_img = os.path.join(input_dir, "img_dir", "val", "organ")
    in_train_ann = os.path.join(input_dir, "ann_dir", "train", "organ")
    in_val_ann = os.path.join(input_dir, "ann_dir", "val", "organ")

    out_train_img = os.path.join(output_dir, "img_dir", "train", "organ")
    out_val_img = os.path.join(output_dir, "img_dir", "val", "organ")
    out_train_ann = os.path.join(output_dir, "ann_dir", "train", "organ")
    out_val_ann = os.path.join(output_dir, "ann_dir", "val", "organ")
    
    ins = [in_train_img, in_val_img, in_train_ann, in_val_ann]
    outs = [out_train_img, out_val_img, out_train_ann, out_val_ann]

    os.makedirs(out_train_img, exist_ok=True)
    os.makedirs(out_val_img, exist_ok=True)
    os.makedirs(out_train_ann, exist_ok=True)
    os.makedirs(out_val_ann, exist_ok=True)

    for in_folder, out_folder in zip(ins, outs):
        for file in os.listdir(in_folder):
            in_path = os.path.join(in_folder, file)
            out_path = os.path.join(out_folder, file)
            if not os.path.isfile(out_path):
                os.link(in_path, out_path)
            if "ann" in in_path and not "PAS" in in_path:
                file = file.split(".")[0] + "_PAS.png"
                out_path = os.path.join(out_folder, file)
                if not os.path.isfile(out_path):
                    os.link(in_path, os.path.join(out_folder, file))

if "__main__" in __name__:
    input_folder = "/home/mawanda/Documents/HuBMAP/for_mmdetection_multires_512"
    output_folder = "/home/mawanda/Documents/HuBMAP/for_mmdetection_multires_512_w_stain"

    # prepare folder - hardlink of original images so they do not occupy more space
    prepare_folders(input_folder, output_folder)

    # Find all images to be stained
    train_img_folder = os.path.join(output_folder, "img_dir", "train", "organ")
    val_img_folder = os.path.join(output_folder, "img_dir", "val", "organ")
    images = [os.path.join(train_img_folder, file) for file in os.listdir(train_img_folder)]
    images += [os.path.join(val_img_folder, file) for file in os.listdir(val_img_folder)]

    with concurrent.futures.ProcessPoolExecutor(max_workers=os.cpu_count()) as worker:
        _ = list(tqdm(worker.map(stain_img, images), total=len(images)))
    # for image in tqdm(images):
        # stain_img(image)
