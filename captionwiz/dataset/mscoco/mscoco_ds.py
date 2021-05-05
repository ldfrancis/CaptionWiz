from captionwiz.dataset.caption_ds import CaptionDS
from captionwiz.dataset.mscoco.fetcher import obtain_annotations, obtain_images
from captionwiz.utils.constants import MSCOCO, MSCOCO_KARPATHY
from captionwiz.utils.log_utils import logger


class MSCOCO_CaptionDS(CaptionDS):
    """"""

    def __init__(self) -> None:
        name = MSCOCO
        super(MSCOCO_CaptionDS, self).__init__(name=name)

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
