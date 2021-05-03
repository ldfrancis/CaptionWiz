import tensorflow as tf

from captionwiz.model.mlp import MLPEncoder
from captionwiz.model.rnn import RNNDecoder
from captionwiz.utils.tf_utils import replace_with_val_at_ind

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
        name="ShowAttTell",
    ):
        super(ShowAttTell, self).__init__(tokenizer=tokenizer, name=name)
        self.encoder = MLPEncoder(embedding_dim)
        self.decoder = RNNDecoder(embedding_dim, units, vocab_size)
        self.tokenizer = tokenizer
        self.end_id = self.tokenizer.word_index("<end>")
        self.max_length = max_length
        self.eval_loss = tf.Variable([0])  # to be used by scheduler

    def call(self, im_features, hidden, target=None, train=False):
        """ """
        assert (target is None) and (not train) or (not (target is None) and train)
        im_embedding = self.encoder(im_features)
        if train:
            hidden = self.decoder.reset_states(target.shape[0])
            input_ = tf.expand_dims(
                [self.tokenizer.word_index["<start>"]] * target.shape[0], 1
            )
            for i in range(1, target.shape[1]):
                pred, hidden, _ = self.decoder(input_, im_embedding, hidden)

            self.decoder()

    @tf.function
    def train_step(self, im_features, target):
        time_steps = target.shape[1]
        batch_size = target.shape[0]
        hidden = self.decoder.reset_states(batch_size)
        input_ = tf.expand_dims([self.tokenizer.word_index["<start>"]] * batch_size, 1)
        loss = 0

        with tf.GradientTape() as tape:
            im_embedding = self.encoder(im_features)
            for i in range(1, time_steps):
                pred, hidden, _ = self.decoder(input_, im_embedding, hidden)
                loss += self.loss(target[:, i], pred)
                input_ = tf.expand_dims(target[:, i], 1)

        loss_per_step = loss / time_steps
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
        predicted = tf.transpose(tf.zeros((target.shape[0], self.max_length)))

        for i in range(1, self.max_length):
            pred, hidden, att_weights = self.decoder(input_, im_embedding, hidden)
            pred_id = tf.random.categorical(pred, 1)
            input_ = pred_id
            predicted = replace_with_val_at_ind(predicted, i - 1, pred_id)

        predicted = tf.transpose(predicted)

        loss = self.loss(target, predicted[:, : target.shape[1]])

        self.eval_loss.assign(loss)

        return loss, loss / target.shape[1]

    @tf.function
    def infer(self, im_features):
        hidden = self.decoder.reset_states(im_features.shape[0])
        input_ = tf.expand_dims(
            [self.tokenizer.word_index["<start>"]] * im_features.shape[0], 1
        )
        im_embedding = self.encoder(im_features)
        predicted = tf.transpose(tf.zeros((im_features.shape[0], self.max_length)))

        for i in range(1, self.max_length):
            pred, hidden, att_weights = self.decoder(input_, im_embedding, hidden)
            pred_id = tf.random.categorical(pred, 1)
            input_ = pred_id
            predicted = replace_with_val_at_ind(predicted, i - 1, pred_id)

        predicted = tf.transpose(predicted)

        return predicted
