import os
from functools import partial
from pathlib import Path

from .config_utils import get_captionwiz_dir, get_datetime
from .img_utils import load_image
from .type import Dir, FilePath

BASE_DIR: FilePath = Path(os.getcwd())

# TIME
DTIME = get_datetime()

# DATASET ##

# MSCOCO
MSCOCO: str = "mscoco"
MSCOCO_DATA_DIR: Dir = get_captionwiz_dir() / "mscoco"

MSCOCO_ANNOTATIONS_URL: str = (
    "http://images.cocodataset.org/annotations/annotations_trainval2014.zip"
)

MSCOCO_TRAIN_ANNOTATIONS_FILE: FilePath = (
    MSCOCO_DATA_DIR / "annotations/captions_train2014.json"
)
MSCOCO_TRAIN_IMAGES_DIR: Dir = MSCOCO_DATA_DIR / "train2014"
MSCOCO_TRAIN_IMAGES_URL: str = "http://images.cocodataset.org/zips/train2014.zip"

MSCOCO_VAL_ANNOTATIONS_FILE: FilePath = (
    MSCOCO_DATA_DIR / "annotations/captions_val2014.json"
)
MSCOCO_VAL_IMAGES_DIR: Dir = MSCOCO_DATA_DIR / "val2014"
MSCOCO_VAL_IMAGES_URL: str = "http://images.cocodataset.org/zips/val2014.zip"

MSCOCO_TEST_IMAGES_DIR: Dir = MSCOCO_DATA_DIR / "test2014"
MSCOCO_TEST_IMAGES_URL: str = "http://images.cocodataset.org/zips/test2014.zip"

# MSCOCO KARPATHY
MSCOCO_KARPATHY: str = "mscoco-karpathy"

# VIZWIZ
VIZWIZ = "vizwiz"
VIZWIZ_DATA_DIR: FilePath = get_captionwiz_dir() / VIZWIZ

VIZWIZ_TEST_IMAGES_DIR: FilePath = VIZWIZ_DATA_DIR
VIZWIZ_TEST_IMAGES_URL: FilePath = (
    "https://ivc.ischool.utexas.edu/VizWiz_final/images/test.zip"
)
VIZWIZ_TEST_ANNOTATIONS_URL: str = (
    "http://ivc.ischool.utexas.edu/VizWiz_final/caption/AoANet_VizWiz/data/test.json"
)
VIZWIZ_TEST_ANNOTATIONS_FILE: FilePath = VIZWIZ_DATA_DIR / "test.json"

VIZWIZ_TRAIN_ANNOTATIONS_FILE: FilePath = VIZWIZ_DATA_DIR / "train.json"
VIZWIZ_TRAIN_IMAGES_DIR: FilePath = VIZWIZ_DATA_DIR
VIZWIZ_TRAIN_IMAGES_URL: FilePath = (
    "https://ivc.ischool.utexas.edu/VizWiz_final/images/train.zip"
)
VIZWIZ_TRAIN_ANNOTATIONS_URL: str = (
    "http://ivc.ischool.utexas.edu/VizWiz_final/caption/AoANet_VizWiz/data/train.json"
)

VIZWIZ_VAL_ANNOTATIONS_FILE: FilePath = VIZWIZ_DATA_DIR / "val.json"
VIZWIZ_VAL_IMAGES_DIR: FilePath = VIZWIZ_DATA_DIR
VIZWIZ_VAL_IMAGES_URL: str = (
    "https://ivc.ischool.utexas.edu/VizWiz_final/images/val.zip"
)
VIZWIZ_VAL_ANNOTATIONS_URL: str = (
    "http://ivc.ischool.utexas.edu/VizWiz_final/caption/AoANet_VizWiz/data/val.json"
)

# MODEL ##
SHOW_ATT_TELL: str = "show_att_tell"


# EXTRACTOR ##

# extractor names
INCEPTIONV3 = "inceptionv3"

# image loaders
INCEPTIONV3_IMSIZE = (299, 299)
IMAGE_LOADERS = {INCEPTIONV3: partial(load_image, im_size=INCEPTIONV3_IMSIZE)}


# OPTIMISER ##
ADAM = "adam"

# mscoco
MSCOCO_FEATURES_DIR = MSCOCO_DATA_DIR / "features"

# TRAINER ##
CHECKPOINT_DIR = get_captionwiz_dir() / "checkpoints"

# LOGGING ##
LOG_DIR = get_captionwiz_dir() / "logs"
LOG_FILE = LOG_DIR / f"{DTIME}_caption_program.log"

# Tensorboard
TENSORBOARD_LOG_DIR: str = LOG_DIR / "tensorboard"
