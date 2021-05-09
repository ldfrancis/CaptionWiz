from typing import Any, Dict, List, Tuple, Union

import numpy as np
import pandas as pd
from tqdm import tqdm

from captionwiz.dataset.caption_ds import CaptionDS
from captionwiz.dataset.vizwiz.fetcher import obtain_annotations, obtain_images
from captionwiz.utils.constants import VIZWIZ, VIZWIZ_CAPTION_PAIR_DIR
from captionwiz.utils.log_utils import logger


class VIZWIZ_CaptionDS(CaptionDS):
    """ """

    def __init__(self, cfg: Dict) -> None:
        name = VIZWIZ
        self._cfg = cfg
        self.max_length = self._cfg["max_length"]
        super(VIZWIZ_CaptionDS, self).__init__(
            name=name,
            cfg=cfg,
        )

    def analyze(self, word_count_thresh):
        """Analyze the dataset; obtaining the word count, top words, etc"""
        filter = '!"#$%&()*+.,-/:;=?@[]^_`{|}~ '

        word_count_map = {}
        sentence_lengths = []
        sentence_lengths_map = {}

        annotations = obtain_annotations("train")["annotations"]
        annotations += obtain_annotations("val")["annotations"]

        num_annotations = len(annotations)
        num_images = len(list(obtain_images("train").iterdir()))

        for ann in annotations:
            caption = f"<start> {ann['caption']} <end>"
            caption = caption if caption[-1] != "." else caption[:-1]
            words = caption.split(" ")
            length = len(words)
            sentence_lengths += [length]
            sentence_lengths_map[length] = sentence_lengths_map.get(length, 0) + 1

            for word in words:
                word = "".join([w for w in word if not (w in filter)])
                word = word.strip().lower()
                word_count_map[word] = word_count_map.get(word, 0) + 1

        total_words = sum(word_count_map.values())
        sentence_length_mean = np.mean(sentence_lengths)
        sentence_length_max = np.max(sentence_lengths)
        sentence_length_min = np.min(sentence_lengths)

        word_count_tuple = sorted(
            [(count, w) for w, count in word_count_map.items()], reverse=True
        )

        logger.info(f"number of annotations: {num_annotations}")
        logger.info(f"number of images: {num_images}")

        logger.info(f"total_words: {total_words} ")
        logger.info(f"sentence length mean: {sentence_length_mean}")
        logger.info(f"sentence length max: {sentence_length_max}")
        logger.info(f"sentence length min: {sentence_length_min}")

        logger.info(f"top 20 words:\n{word_count_tuple[:20]}")
        logger.info(
            f"num words with counts greater than {word_count_thresh}: "
            f"{len([w for c,w in word_count_tuple if c > word_count_thresh])}"
        )

    def create_image_caption_pairs(self):
        """Method sets the im_to_caption and image_caption_pairs attributes for the
        train and val splits. The list of test images, test_images, is also set
        """

        def _img_id_to_filename(
            images: List[Dict[str, Any]], image_id: int
        ) -> Union[str, None]:
            for i in range(len(images)):
                im1 = images[i]
                im2 = images[-i]

                if im1["id"] == image_id:
                    return im1["file_name"]
                elif im2["id"] == image_id:
                    return im2["file_name"]
            return None

        logger.info(f"Starting to create image-caption pairs for dataset {self.name}")
        for split in ["train", "val"]:
            csv_file = VIZWIZ_CAPTION_PAIR_DIR / f"{split}.csv"
            if csv_file.exists():
                continue
            annotations = obtain_annotations(split)
            image_dir = obtain_images(split)
            images = annotations["images"]
            image_caption_pairs: List[Tuple[str, str]] = []
            for ann in tqdm(annotations["annotations"]):
                caption = f"<start> {ann['caption']} <end>"
                image_path = (
                    str(image_dir.absolute())
                    + f'/{_img_id_to_filename(images, ann["image_id"])}'
                )
                image_caption_pairs += [(image_path, caption)]
            df = pd.DataFrame(
                {
                    "img": [im for im, _ in image_caption_pairs],
                    "cap": [cap for _, cap in image_caption_pairs],
                }
            )
            df.to_csv(VIZWIZ_CAPTION_PAIR_DIR / f"{split}.csv", index=False)

            logger.info(f"Done creating {split} image-caption pairs")

        train_image_caption = pd.read_csv(VIZWIZ_CAPTION_PAIR_DIR / "train.csv")
        val_image_caption = pd.read_csv(VIZWIZ_CAPTION_PAIR_DIR / "val.csv")
        self.train_image_caption_pairs = [
            (im, cap)
            for im, cap in zip(train_image_caption["img"], train_image_caption["cap"])
        ]
        self.val_image_caption_pairs = [
            (im, cap)
            for im, cap in zip(val_image_caption["img"], val_image_caption["cap"])
        ]

        # test split
        self._test_image_dir = obtain_images("test")
        self.test_images = [str(f.absolute()) for f in self._test_image_dir.iterdir()]
        logger.info("Created test images")
