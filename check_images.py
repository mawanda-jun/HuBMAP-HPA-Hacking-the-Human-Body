import shutil
import cv2
import os
import concurrent.futures
from tqdm import tqdm

# Keep some images for validation. I've selected those by looking 
kidney = [62, 164, 5099, 5102, 17422, 18426, 18777, 27298, 27468, 30876]
prostate = [435, 4658, 4944, 7902, 8227, 14396, 22995, 29307, 32412]
largeintestine = [10651, 11890, 12471, 25298, 25689, 31799]
spleen = [10703, 12026, 19377, 19997, 32231]
lung = [1878, 4301, 5086, 16564, 23252]

val_imgs = [*kidney, *prostate, *largeintestine, *spleen, *lung]


def check_img(in_path):
    try:
        img = cv2.imread(in_path)
        assert len(img.shape) == 3
    except Exception as e:
        print(f"Exception was {e}, deleting image.")
        os.remove(in_path)
        ann_path = in_path.replace("img", "ann")
        os.remove(ann_path)

def move_to_val(in_path):
    img_id = int(os.path.basename(in_path).split("_")[0])
    if img_id in val_imgs:
        out_path = in_path.replace("train", "val")
        shutil.move(in_path, out_path)
        # Move annotation
        ann_path = in_path.replace("img", "ann")
        out_ann = ann_path.replace("train", "val")
        shutil.move(ann_path, out_ann)


def check_train_val(input_dir):
    in_train_img = os.path.join(input_dir, "img_dir", "train", "organ")
    in_val_img = os.path.join(input_dir, "img_dir", "val", "organ")
    
    ins = [in_train_img, in_val_img]

    in_paths = []
    for in_folder in ins:
        for file in os.listdir(in_folder):
            in_path = os.path.join(in_folder, file)
            in_paths.append(in_path)

    with concurrent.futures.ProcessPoolExecutor(max_workers=os.cpu_count()) as worker:
        _ = list(tqdm(worker.map(check_img, in_paths), total=len(in_paths)))


def move_train_to_val(input_dir):
    in_train_img = os.path.join(input_dir, "img_dir", "train", "organ")
    
    paths = []
    for file in os.listdir(in_train_img):
        in_path = os.path.join(in_train_img, file)
        paths.append(in_path)

    with concurrent.futures.ProcessPoolExecutor(max_workers=os.cpu_count()) as worker:
        _ = list(tqdm(worker.map(move_to_val, paths), total=len(paths)))


if "__main__" in __name__:
    move_train_to_val(
        "/home/mawanda/Documents/HuBMAP/for_mmdetection_multires_512_w_stain_inverted"
    )
    
