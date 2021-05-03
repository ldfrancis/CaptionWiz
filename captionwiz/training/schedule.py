import tensorflow as tf
from tensorflow.keras.metrics import Mean
from tensorflow.keras.optimizers.schedules import LearningRateSchedule


class CaptionLrSchedule(LearningRateSchedule):
    def __init__(self, model, decay, initial_lr, patience):
        self._model = model
        self._decay = tf.cast(decay, tf.float32)
        self._metric = Mean()
        self._initial_lr = initial_lr
        self._state = tf.Variable(0, tf.int32)
        self._patience = tf.Variable(0, tf.int32)
        self._max_patience = tf.cast(patience, tf.int32)
        self._lr = tf.Variable(0, tf.float32)

    def __call__(self, step):
        metric = self._model.eval_loss
        self._metric.update_state(metric)
        if tf.greater(metric, self._metric.result()):
            self._patience.assign_add(1)
            if tf.greater(self._patience, self._max_patience):
                self._patience.assign(0)
                if tf.equal(self._state, 0):
                    self._state.assign(1)
                    self._lr.assign(0.99 * self._lr)
                else:
                    self._lr.assign(0.99 * self._lr)

        if tf.equal(self._state, 0):
            self._lr.assign(1.01 * self._lr)

        return self._lr
