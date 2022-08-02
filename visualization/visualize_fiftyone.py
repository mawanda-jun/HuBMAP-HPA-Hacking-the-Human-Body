import fiftyone as fo
import os


def create_dataset():
    # A name for the dataset
    name = "my-dataset"

    # The directory containing the dataset to import

    dataset_dir = "/home/mawanda/Documents/HuBMAP/"
    ann_path = os.path.join(dataset_dir, "val_coco.json")
    imgs_path = os.path.join(dataset_dir, "train_images_png")

    # The type of the dataset being imported
    dataset_type = fo.types.COCODetectionDataset

    dataset = fo.Dataset.from_dir(
        dataset_dir=dataset_dir,
        dataset_type=dataset_type,
        name=name,
        data_path=imgs_path,
        labels_path=ann_path,
        label_types=["detections", "segmentations"]
    )

    return dataset

if "__main__" in __name__:
    dataset = create_dataset()
    session = fo.launch_app(dataset, remote=True)
    session.wait()