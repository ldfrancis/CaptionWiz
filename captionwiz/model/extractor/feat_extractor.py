from pathlib import Path
from typing import Tuple

import numpy as np
import tensorflow as tf

from captionwiz.dataset.caption_ds import CaptionDS
from captionwiz.model.extractor.base_extractor import get_inceptionV3
from captionwiz.utils.constants import (
    IMAGE_LOADERS,
    INCEPTIONV3,
    MSCOCO,
    MSCOCO_FEATURES_DIR,
)
from captionwiz.utils.data_utils import dataloader_from_img_paths
from captionwiz.utils.type import FilePath, Tensor


class FeatureExtractor:
    """Extracts features from an images in a dataset (if given)

    available extractors:
        - InceptionV3

    """

    def __init__(
        self, extractor_name: str, dataset: CaptionDS = None, batch_size: int = 16
    ):
        self._dataset = dataset

        assert extractor_name in _extractors.keys(), (
            f"extractor {extractor_name} is not "
            f"available choose either of these {_extractors.keys()}"
        )

        self._extractor_name = extractor_name
        self.extraction_model = _extractors[extractor_name]

        self._batch_size = batch_size

        if self._dataset:
            self._image_paths = list(self._dataset.im_to_captions.keys())
            self._image_paths = sorted(set(self._image_paths))
            self.im_to_features = {im: None for im in self._image_paths}
            self.run_on_images(batch_size)

    def run_on_images(self, batch_size: int = 16):
        """Runs the extractor on all images in the dataset (if available) and stores results
        in a mapping, image_features
        """
        assert (
            self._dataset is not None
        ), "No dataset is give, cannot run extractor on images"

        imageloader = IMAGE_LOADERS[self._extractor_name]
        ds = dataloader_from_img_paths(self._image_paths, imageloader, batch_size)

        self._features_dir = _captionds_features_dir[self._dataset.name]

        for im, im_pth in ds:
            for m, p in zip(im, im_pth):
                features = self.extraction_model(m)
                filepath = p.numpy().decode("utf-8")
                filename = Path(filepath).name
                features_filepath = (
                    _captionds_features_dir[self._dataset.name] / filename
                )
                self.im_to_features[filepath] = features_filepath
                to_save = {
                    fn: f.numpy()
                    for f, fn in zip(
                        features, _extractors_features[self._extractor_name]
                    )
                }
                np.savez_compressed(filepath, **to_save)

    def extract_from_file(self, im_path: FilePath) -> Tuple[Tensor, ...]:
        """Extract the features for an image with the given path

        Args:
            im_path(FilePath): The path to the image

        Returns:
            Tuple[Tensor]: A tuple of Tensors containing the extracted features
        """

        def _load_img_and_batch(p):
            image_loader = IMAGE_LOADERS[self._extractor_name]
            img = image_loader(im_path)
            batched_img = tf.expand_dims(img, 0)
            return batched_img

        if self._dataset:
            feature_filepath = self.im_to_features[im_path]
            features_cmprsd = np.load(feature_filepath)
            features = tuple(
                [features_cmprsd[k] for k in _extractors_features[self._extractor_name]]
            )
        else:
            batched_features = self.extraction_model(_load_img_and_batch(im_path))
            features = tf.squeeze(batched_features, axis=0)

        return features

    def create_image_caption_dataloader(
        self,
        buffer_size: int = 1000,
        batch_size: int = 16,
        shuffle: bool = True,
        split="train",
    ):
        """Creates a dataloader that loads image features and captions pairs

        Returns:
            tf.data.Dataset: A dataset that iterates over the image features and
            captions
        """
        assert self._dataset is not None
        assert split in ["train", "test", "val"]

        if split == "train":
            im_paths = self._dataset.train_im_paths
            captions = self._dataset.train_captions
        elif split == "test":
            im_paths = self._dataset.test_im_paths
            captions = self._dataset.test_captions
        else:
            im_paths = self._dataset.val_im_paths
            captions = self._dataset.val_captions

        # handle shuffle
        shuffle = False if (split == "test") else True

        dataset = tf.data.Dataset.from_tensor_slices((im_paths, captions))

        def _map_fn(img_pth, cap):
            feat = self.extract_from_file(img_pth)
            return (*feat, cap)

        dataset = dataset.map(
            lambda pth, cap: tf.numpy_function(
                _map_fn,
                [pth, cap],
                [tf.float32, tf.int32],
                num_parallel_calls=tf.data.AUTOTUNE,
            )
        )
        dataset = dataset.shuffle(buffer_size) if shuffle else dataset
        dataset = dataset.batch(batch_size)
        dataset = dataset.prefetch(buffer_size=tf.data.AUTOTUNE)

        return dataset


_extractors = {
    INCEPTIONV3: get_inceptionV3(),
}

_extractors_features = {
    INCEPTIONV3: ("feat"),
}

_captionds_features_dir = {
    MSCOCO: MSCOCO_FEATURES_DIR,
}
