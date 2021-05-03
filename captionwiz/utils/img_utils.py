from typing import Tuple

import tensorflow as tf
from tensorflow.keras.applications.inception_v3 import preprocess_input

from captionwiz.utils.type import FilePath


def load_image(img_path: FilePath, im_size: Tuple[int, int] = None) -> tf.Tensor:
    file = tf.io.read_file(img_path)
    img = tf.image.decode_jpeg(file, channels=3)
    if im_size:
        resized_img = tf.image.resize(img)
    else:
        resized_img = img
    resized_img = preprocess_input(resized_img)
    return resized_img, img
