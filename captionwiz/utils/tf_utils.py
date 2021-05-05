import tensorflow as tf

from captionwiz.utils.type import Tensor


def replace_with_val_at_ind(tensor: Tensor, ind: int, val: Tensor):
    ones = tf.ones_like(tensor, dtype=tensor.dtype)
    ind_range = tf.range(tensor.shape[0])
    ind_range = tf.reshape(ind_range, (tensor.shape[0], *[1] * len(tensor.shape[1:])))
    ind_mask = tf.cast(ind_range == ind, tensor.dtype)
    ind_one = ones * ind_mask
    ind_zero = tf.cast(tf.logical_not(tf.cast(ind_one, bool)), tensor.dtype)
    new_tensor = ind_zero * tensor + ind_one * tf.squeeze(val)
    return new_tensor
