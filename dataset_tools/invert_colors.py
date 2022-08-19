import cv2
import os
import concurrent.futures
from tqdm import tqdm

def invert_color(in_path, out_path):
    if "ann" in in_path:
        os.link(in_path, out_path)
    else:
        img = cv2.imread(in_path)
        img = 255-img
        cv2.imwrite(out_path, img)

def prepare_folder(input_dir, output_dir):
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

    in_paths = []
    out_paths = []
    for in_folder, out_folder in zip(ins, outs):
        for file in os.listdir(in_folder):
            in_path = os.path.join(in_folder, file)
            in_paths.append(in_path)
            out_path = os.path.join(out_folder, file)
            out_paths.append(out_path)

    with concurrent.futures.ProcessPoolExecutor(max_workers=os.cpu_count()) as worker:
        _ = list(tqdm(worker.map(invert_color, in_paths, out_paths), total=len(in_paths)))


if "__main__" in __name__:
    folder_path = "/home/mawanda/Documents/HuBMAP/for_mmdetection_resized_5000_w_stain"
    prepare_folder(
        folder_path,
        folder_path + "_inverted"
    )
    
