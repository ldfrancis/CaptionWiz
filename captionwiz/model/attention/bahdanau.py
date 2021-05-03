from typing import Tuple

import tensorflow as tf
from tensorflow.keras.layers import Dense

from captionwiz.utils.type import Tensor


class BahdanauAttention(tf.keras.Model):
    """BahdanauAttention"""

    def __init__(self, units: int):
        super(BahdanauAttention, self).__init__()
        self.W1 = Dense(units)
        self.W2 = Dense(units)
        self.V = Dense(1)

    def call(self, features: Tensor, hidden: Tensor) -> Tuple[Tensor, Tensor]:
        """Forward pass throught attention model to generate context vector and attention
            weights

        Args:
            features (Tensor): features from encoder. shape (B, F, embedding_dim)
            hidden (Tensor): Decoder hidden state. shape (B, hidden_size)
        """
        hidden = tf.expand_dims(hidden, 1)
        attention_hidden = tf.nn.tanh(self.W1(features) + self.W2(hidden))
        attention_weights = tf.nn.softmax(self.V(attention_hidden), axis=1)

        context_vector = attention_weights * features
        context_vector = tf.reduce_sum(context_vector, axis=1)

        return context_vector, attention_weights
