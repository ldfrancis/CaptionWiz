import json
import os
from typing import Any, Dict

from tensorflow.keras.utils import get_file

from captionwiz.utils.constants import (
    MSCOCO_ANNOTATIONS_URL,
    MSCOCO_DATA_DIR,
    MSCOCO_TEST_IMAGES_DIR,
    MSCOCO_TEST_IMAGES_URL,
    MSCOCO_TRAIN_ANNOTATIONS_FILE,
    MSCOCO_TRAIN_IMAGES_DIR,
    MSCOCO_TRAIN_IMAGES_URL,
    MSCOCO_VAL_ANNOTATIONS_FILE,
    MSCOCO_VAL_IMAGES_DIR,
    MSCOCO_VAL_IMAGES_URL,
)
from captionwiz.utils.type import Dir


def obtain_annotations(split="train") -> Dict[str, Any]:
    """Obtain the annotations file for mscoco train and val splits"""
    assert split in ["train", "val"], "split can either be train or val"
    annotations_file = {
        "train": MSCOCO_TRAIN_ANNOTATIONS_FILE,
        "val": MSCOCO_VAL_ANNOTATIONS_FILE,
    }[split]
    if annotations_file.exists():
        annotations = json.load(open(annotations_file, "r"))
    else:
        # download annotations file
        download_annotations_file()
        annotations = json.load(open(annotations_file, "r"))

    return annotations


def download_annotations_file() -> None:
    """Downloads the mscoco annotations file, train n val splits"""

    annotations_zip_file = get_file(
        "annotations.zip",
        cache_subdir=MSCOCO_DATA_DIR,
        origin=MSCOCO_ANNOTATIONS_URL,
        extract=True,
    )
    os.remove(annotations_zip_file)


def download_images(split="train") -> None:
    """Downloads the mscoco images for the train and val splits"""
    assert split in ["train", "val", "test"], "split can either be train, val, or test"
    url = {
        "train": MSCOCO_TRAIN_IMAGES_URL,
        "val": MSCOCO_VAL_IMAGES_URL,
        "test": MSCOCO_TEST_IMAGES_URL,
    }[split]
    images_zip_file = get_file(
        "images.zip", cache_subdir=MSCOCO_DATA_DIR, origin=url, extract=True
    )
    os.remove(images_zip_file)


def obtain_images(split="train") -> Dir:
    """Returns the directory containing the images"""
    assert split in ["train", "val", "test"], "split can either be train, val, or test"
    directory = {
        "train": MSCOCO_TRAIN_IMAGES_DIR,
        "val": MSCOCO_VAL_IMAGES_DIR,
        "test": MSCOCO_TEST_IMAGES_DIR,
    }[split]
    if not directory.exists():
        download_images(split)
    assert directory.exists(), f"No directory for images. {directory} does not exist"
    return directory
