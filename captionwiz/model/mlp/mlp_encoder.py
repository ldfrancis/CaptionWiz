from typing import Any, Tuple

import tensorflow as tf
from tensorflow.keras.layers import Activation, BatchNormalization, Dense
from tensorflow.keras.models import Sequential

from captionwiz.utils.type import Tensor


class MLPEncoder(tf.keras.Model):
    """Encodes image features into an embedding of specified size"""

    def __init__(self, embedding_dim: int, layers: Tuple[Any] = None):
        super(MLPEncoder, self).__init__()
        if not layers:
            self.embedder = Dense(embedding_dim)
        else:
            lyrs = []
            for i, lval in enumerate(layers):
                if isinstance(lval, int):
                    lyrs += [Dense(lval)]
                elif isinstance(lval, str):
                    if lval == "bn":
                        lyrs += [BatchNormalization(axis=-1)]
                    else:
                        # has to be an activation
                        act = Activation(lval)
                        lyrs += [act]

            lyrs += [Dense(embedding_dim)]
            self.embedder = Sequential(lyrs)

    def call(self, x: Tensor) -> Tensor:
        """Forward pass through features encoder

        Args:
            x (Tensor): input tensor

        Returns:
            Tensor: output tensor
        """
        x = self.embedder(x)
        x = tf.nn.relu(x)

        return x
