import tensorflow as tf
from tensorflow.keras.metrics import Mean
from tensorflow.keras.optimizers.schedules import LearningRateSchedule


class CaptionLrSchedule(LearningRateSchedule):
    def __init__(self, model, decay, max_lr, patience):
        self._model = model
        self._decay = tf.cast(decay, name="decay", dtype=tf.float32)
        self._metric = Mean()
        self._metric.update_state(1e4)
        self._max_lr = max_lr
        self._state = tf.Variable(0, name="state", dtype=tf.int32)
        self._patience = tf.Variable(0, name="patience", dtype=tf.int32)
        self._max_patience = tf.cast(patience, name="max_patience", dtype=tf.int32)
        self._lr = tf.Variable(1e-9, name="_lr", dtype=tf.float32)

    def __call__(self, step):
        metric = self._model.train_loss
        self._metric.update_state(metric)

        x = tf.identity(1)

        def x_fn(x):
            return x

        def _decay_lr():
            self._patience.assign(0)
            tf.cond(tf.equal(self._state, 0), _change_state_n_decay, _only_decay)
            return x

        def _change_state_n_decay():
            self._state.assign(1)
            self._lr.assign(0.99 * self._lr)
            return x

        def _only_decay():
            self._lr.assign(0.99 * self._lr)
            return x

        def _lr_warmup():
            self._lr.assign(1.1 * self._lr)
            return x

        def _should_decay_lr():
            self._patience.assign_add(1)
            tf.cond(tf.greater(self._patience, self._max_patience), _decay_lr, x_fn)
            return x

        tf.cond(tf.greater(metric, self._metric.result()), _should_decay_lr, x_fn)
        tf.cond(tf.equal(self._state, 0), _lr_warmup, x_fn)

        self._lr = tf.minimum(self._lr, self.max_lr)

        return self._lr
