import tensorflow as tf
from tensorflow.keras.optimizers.schedules import LearningRateSchedule


class CaptionLrSchedule(LearningRateSchedule):
    def __init__(self, model, decay, lr, patience):
        self._model = model
        self._decay = tf.cast(decay, name="decay", dtype=tf.float32)
        self._patience = tf.Variable(0, name="patience", dtype=tf.int32)
        self._max_patience = tf.cast(patience, name="max_patience", dtype=tf.int32)
        self._lr = tf.Variable(lr, name="_lr", dtype=tf.float32)
        self.prev_loss = tf.Variable(0, name="prev_loss", dtype=tf.float32)

    def __call__(self, step):
        metric = self._model.eval_loss

        x = tf.identity(1)

        def x_fn():
            return x

        def _decay_lr():
            self._patience.assign(0)
            self._lr.assign(0.99 * self._lr)
            return x

        def _should_decay_lr():
            self._patience.assign_add(1)
            tf.cond(tf.greater(self._patience, self._max_patience), _decay_lr, x_fn)
            return x

        tf.cond(tf.greater(metric, self.prev_loss), _should_decay_lr, x_fn)

        self.prev_loss.assign(metric)

        return self._lr
