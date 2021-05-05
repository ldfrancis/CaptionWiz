from typing import Tuple

import tensorflow as tf

from captionwiz.utils.type import FilePath


def load_image(img_path: FilePath, im_size: Tuple[int, int] = None) -> tf.Tensor:
    file = tf.io.read_file(img_path)
    img = tf.image.decode_jpeg(file, channels=3)
    if im_size:
        resized_img = tf.image.resize(img, im_size)
    else:
        resized_img = img

    return resized_img, img_path
