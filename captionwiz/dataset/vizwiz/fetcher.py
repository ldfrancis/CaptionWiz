import json
import os
from typing import Any, Dict

from tensorflow.keras.utils import get_file

from captionwiz.utils.constants import (
    VIZWIZ_DATA_DIR,
    VIZWIZ_TEST_ANNOTATIONS_FILE,
    VIZWIZ_TEST_ANNOTATIONS_URL,
    VIZWIZ_TEST_IMAGES_DIR,
    VIZWIZ_TEST_IMAGES_URL,
    VIZWIZ_TRAIN_ANNOTATIONS_FILE,
    VIZWIZ_TRAIN_ANNOTATIONS_URL,
    VIZWIZ_TRAIN_IMAGES_DIR,
    VIZWIZ_TRAIN_IMAGES_URL,
    VIZWIZ_VAL_ANNOTATIONS_FILE,
    VIZWIZ_VAL_ANNOTATIONS_URL,
    VIZWIZ_VAL_IMAGES_DIR,
    VIZWIZ_VAL_IMAGES_URL,
)
from captionwiz.utils.type import Dir


def obtain_annotations(split="train") -> Dict[str, Any]:
    """Obtain the annotations file for vizwiz train and val splits"""
    assert split in ["train", "val", "test"], "split can either be train, val or test"
    annotations_file = {
        "train": VIZWIZ_TRAIN_ANNOTATIONS_FILE,
        "val": VIZWIZ_VAL_ANNOTATIONS_FILE,
        "test": VIZWIZ_TEST_ANNOTATIONS_FILE,
    }[split]
    if annotations_file.exists():
        annotations = json.load(open(annotations_file, "r"))
    else:
        # download annotations file
        download_annotations_file(split)
        annotations = json.load(open(annotations_file, "r"))

    return annotations


def download_annotations_file(split="train") -> None:
    """Downloads the vizwiz annotations file, train n val splits"""

    url = {
        "train": VIZWIZ_TRAIN_ANNOTATIONS_URL,
        "val": VIZWIZ_VAL_ANNOTATIONS_URL,
        "test": VIZWIZ_TEST_ANNOTATIONS_URL,
    }[split]

    get_file(
        f"{split}.json",
        cache_subdir=VIZWIZ_DATA_DIR,
        origin=url,
        extract=True,
    )


def download_images(split="train") -> None:
    """Downloads the vizwiz images for the train and val splits"""
    assert split in ["train", "val", "test"], "split can either be train, val, or test"
    url = {
        "train": VIZWIZ_TRAIN_IMAGES_URL,
        "val": VIZWIZ_VAL_IMAGES_URL,
        "test": VIZWIZ_TEST_IMAGES_URL,
    }[split]
    images_zip_file = get_file(
        f"{split}.zip", cache_subdir=VIZWIZ_DATA_DIR, origin=url, extract=True
    )
    os.remove(images_zip_file)


def obtain_images(split="train") -> Dir:
    """Returns the directory containing the images"""
    assert split in ["train", "val", "test"], "split can either be train, val, or test"
    directory = {
        "train": VIZWIZ_TRAIN_IMAGES_DIR,
        "val": VIZWIZ_VAL_IMAGES_DIR,
        "test": VIZWIZ_TEST_IMAGES_DIR,
    }[split]
    if not directory.exists():
        download_images(split)
    assert directory.exists(), f"No directory for images. {directory} does not exist"
    return directory
