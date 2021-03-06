from typing import Tuple

import tensorflow as tf

from captionwiz.model.mlp import MLPEncoder
from captionwiz.model.rnn import RNNDecoder
from captionwiz.utils.constants import SHOW_ATT_TELL
from captionwiz.utils.tf_utils import replace_with_val_at_ind
from captionwiz.utils.type import Tensor

from .caption_model import CaptionModel


class ShowAttTell(CaptionModel):
    """An Image captioning model similar to show, attend and tell"""

    def __init__(
        self,
        embedding_dim,
        units,
        vocab_size,
        tokenizer,
        max_length,
    ):
        super(ShowAttTell, self).__init__(
            embedding_dim, units, vocab_size, tokenizer, max_length, name=SHOW_ATT_TELL
        )

        self.encoder = MLPEncoder(embedding_dim)
        self.decoder = RNNDecoder(embedding_dim, units, vocab_size)
        self.tokenizer = tokenizer
        self.end_id = self.tokenizer.word_index["<end>"]
        self.max_length = max_length

    def call(self, im_features) -> Tensor:
        """ """
        return self.infer(im_features)

    @tf.function
    def train_step(self, im_features: Tensor, target: Tensor) -> Tuple[Tensor, Tensor]:
        hidden = self.decoder.reset_states(target.shape[0])
        input_ = tf.expand_dims(
            [self.tokenizer.word_index["<start>"]] * target.shape[0], 1
        )
        loss = 0.0

        with tf.GradientTape() as tape:
            im_embedding = self.encoder(im_features)
            for i in tf.range(1, target.shape[1]):
                pred, hidden, _ = self.decoder(input_, im_embedding, hidden)
                loss += self.loss(target[:, i], pred)
                input_ = tf.expand_dims(target[:, i], 1)

        loss_per_step = loss / target.shape[1]
        vars_ = self.trainable_variables
        grads = tape.gradient(loss, vars_)
        self.optimizer.apply_gradients(zip(grads, vars_))

        return loss, loss_per_step

    @tf.function
    def eval_step(self, im_features, target):
        hidden = self.decoder.reset_states(target.shape[0])
        input_ = tf.expand_dims(
            [self.tokenizer.word_index["<start>"]] * target.shape[0], 1
        )
        im_embedding = self.encoder(im_features)
        loss = 0.0

        for i in tf.range(1, self.max_length):
            pred, hidden, att_weights = self.decoder(input_, im_embedding, hidden)
            pred_id = tf.argmax(pred, axis=-1)
            pred_id = tf.reshape(pred_id, (pred_id.shape[0], 1))
            loss += self.loss(target[:, i], pred)
            input_ = tf.cast(pred_id, tf.int32)

        return loss, loss / target.shape[1]

    @tf.function
    def infer(self, im_features):
        hidden = self.decoder.reset_states(im_features.shape[0])
        input_ = tf.expand_dims(
            [self.tokenizer.word_index["<start>"]] * im_features.shape[0], 1
        )
        im_embedding = self.encoder(im_features)
        predicted = tf.transpose(
            tf.zeros((im_features.shape[0], self.max_length), dtype=tf.int64)
        )

        for i in tf.range(self.max_length):
            pred, hidden, att_weights = self.decoder(input_, im_embedding, hidden)
            pred_id = tf.argmax(pred, axis=-1)
            pred_id = tf.reshape(pred_id, (pred_id.shape[0], 1))
            input_ = pred_id
            predicted = replace_with_val_at_ind(predicted, i, pred_id)
            input_ = tf.cast(pred_id, tf.int32)

        predicted = tf.transpose(predicted)

        return predicted
