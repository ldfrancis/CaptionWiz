from typing import Callable, List, Tuple

import tensorflow as tf

from captionwiz.utils.type import FilePath


def dataloader_from_img_paths(
    img_paths: List[str],
    img_loader: Callable[[FilePath], Tuple[tf.Tensor, FilePath]],
    batch_size: int = 16,
) -> tf.data.Dataset:
    img_ds = tf.data.Dataset.from_tensor_slices(img_paths)
    img_ds = img_ds.map(img_loader, num_parallel_calls=tf.data.AUTOTUNE).batch(
        batch_size
    )
    return img_ds
