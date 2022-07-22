import cv2
import os

def invert_color(img_path):
    img = cv2.imread(img_path)
    img = 255-img
    cv2.imwrite(img_path, img)

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
