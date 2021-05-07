import numpy as np

from captionwiz.dataset.caption_ds import CaptionDS
from captionwiz.dataset.mscoco.fetcher import obtain_annotations, obtain_images
from captionwiz.utils.constants import MSCOCO, MSCOCO_KARPATHY
from captionwiz.utils.log_utils import logger


class MSCOCO_CaptionDS(CaptionDS):
    """"""

    def __init__(self, cfg) -> None:
        name = MSCOCO
        self._cfg = cfg
        self.max_length = self._cfg["max_length"]
        super(MSCOCO_CaptionDS, self).__init__(
            name=name, top_k_words=cfg["top_k_words"]
        )

    def analyze(self, word_count_thresh):
        """Analyze the dataset; obtaining the word count, top words, etc"""
        filter = '!"#$%&()*+.,-/:;=?@[]^_`{|}~ '

        word_count_map = {}
        sentence_lengths = []
        sentence_lengths_map = {}

        annotations = obtain_annotations("train")["annotations"]

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
                word = word.strip()
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
        logger.info(f"Starting to create image-caption pairs for dataset {self.name}")
        # train split
        self._train_annotations = obtain_annotations("train")
        self._train_image_dir = obtain_images("train")

        for ann in self._train_annotations["annotations"]:
            caption = f"<start> {ann['caption']} <end>"
            image_path = (
                str(self._train_image_dir.absolute())
                + "/COCO_train2014_"
                + "%012d.jpg" % (ann["image_id"])
            )
            self.train_im_to_caption[image_path].append(caption)
            self.train_image_caption_pairs += [(image_path, caption)]
        logger.info("Done creating train image-caption pairs")

        # val split
        self._val_annotations = obtain_annotations("val")
        self._val_image_dir = obtain_images("val")

        for ann in self._val_annotations["annotations"]:
            caption = f"<start> {ann['caption']} <end>"
            image_path = (
                str(self._val_image_dir.absolute())
                + "/COCO_val2014_"
                + "%012d.jpg" % (ann["image_id"])
            )
            self.val_im_to_caption[image_path].append(caption)
            self.val_image_caption_pairs += [(image_path, caption)]
        logger.info("Done creating val image-caption pairs")

        # test split
        self._test_image_dir = obtain_images("test")
        self.test_images = [str(f.absolute()) for f in self._test_image_dir.iterdir()]
        logger.info("Created test images")


class MSCOCO_Karpathy_CaptionDS(MSCOCO_CaptionDS):
    def __init__(self):
        super(MSCOCO_Karpathy_CaptionDS, self).__init__()
        self.name = MSCOCO_KARPATHY
