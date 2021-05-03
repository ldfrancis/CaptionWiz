import tensorflow as tf
from tensorflow.keras.losses import SparseCategoricalCrossentropy

from captionwiz.utils.type import Tensor

caption_xe = SparseCategoricalCrossentropy(from_logits=True, reduction="none")


def caption_xe_loss(real: Tensor, pred: Tensor) -> Tensor:
    mask = tf.math.logical_not(tf.math.equal(real, 0))
    loss = caption_xe(real, pred)
    mask = tf.cast(mask, dtype=loss.dtype)
    loss *= mask
    loss = tf.reduce_mean(loss)
    return loss
