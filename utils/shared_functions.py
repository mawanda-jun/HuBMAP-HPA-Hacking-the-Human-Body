from pathlib import Path
from typing import Dict

from utils.generic_functions import load_yaml


def load_validation_images(config_path: Path):
    config: Dict = load_yaml(config_path)

    # Keep some images for validation. I've selected those by looking
    kidney = config["kidney"]
    prostate = config["prostate"]
    largeintestine = config["largeintestine"]
    spleen = config["spleen"]
    lung = config["lung"]

    return [*kidney, *prostate, *largeintestine, *spleen, *lung]
