import os
import argparse
import torch

def main(ckpt_path):
    dirname = os.path.dirname(ckpt_path).split(os.sep)[-1]
    iter = int(os.path.basename(ckpt_path).replace("iter_", "").replace(".pth", ""))
    names = dirname.split("_")
    mode = names[1]
    model = names[2]
    aug = "_".join(names[3:])
    ckpt = torch.load(ckpt_path, map_location='cpu')
    _ = ckpt.pop('optimizer')
    new_path = os.path.join(os.path.dirname(ckpt_path), f"{mode}_{model}_{iter}_{aug}.pth")
    torch.save(ckpt, new_path)

if "__main__" in __name__:
    parser = argparse.ArgumentParser()
    parser.add_argument("--path")
    args = parser.parse_args()
    main(args.path)
    
