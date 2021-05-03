from typing import Tuple

import tensorflow as tf
from tensorflow.keras.layers import GRU, Dense, Embedding

from captionwiz.model.attention import BahdanauAttention
from captionwiz.utils.type import Tensor


class RNNDecoder(tf.keras.Model):
    """Decoder that generates captions from an image embedding using attention"""

    def __init__(self, embedding_dim, units, vocab_size, rnn_type: str = "gru"):
        super(RNNDecoder, self).__init__()
        assert rnn_type in ["gru", "lstm", "rnn"]
        self._units = units
        self.embedding = Embedding(vocab_size, embedding_dim)
        self.rnn = _rnn_to_use[rnn_type](units)
        self.fc1 = Dense(units)
        self.fc2 = Dense(vocab_size)
        self.attention = BahdanauAttention(units)

    def call(
        self, x: Tensor, features: Tensor, hidden: Tensor
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """"""
        context_vector, attention_weights = self.attention(features, hidden)
        x = self.embedding(x)
        x = tf.concat([tf.expand_dims(context_vector, 1), x], axis=-1)
        output, state = self.rnn(x)
        x = self.fc1(output)
        x = tf.reshape(x, (-1, x.shape[2]))
        x = self.fc2(x)

        return x, state, attention_weights

    def reset_states(self, batch_size: int):
        """Resets the rnn state to zeros

        Args:
            batch_size (int): batch size for the input Tensor
        """
        return tf.zeros((batch_size, self._units))


_rnn_to_use = {
    "gru": lambda units: GRU(
        units,
        return_sequences=True,
        return_state=True,
        recurrent_initializer="glorot_uniform",
    ),
}
