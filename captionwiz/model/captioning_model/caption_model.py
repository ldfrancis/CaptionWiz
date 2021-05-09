from abc import abstractmethod
from typing import Dict, Union

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Optimizer
from tensorflow.keras.optimizers import get as get_optim

from captionwiz.utils.type import ImageFeatures, Loss, Tensor


class CaptionModel(Model):
    """"""

    def __init__(
        self,
        embedding_dim,
        units,
        vocab_size,
        tokenizer,
        max_length,
        name,
    ):
        super(CaptionModel, self).__init__(name=name)
        self.tokenizer = tokenizer
        self._embedding_dim = embedding_dim
        self._units = units
        self.vocab_size = vocab_size
        self._max_length = max_length
        self.eval_loss = tf.Variable(
            1e20, name="eval_loss", trainable=False, dtype=tf.float32
        )  # to be used by scheduler

    def optim_prep(self, optimizer: Union[str, Optimizer], loss: Loss):
        if isinstance(optimizer, str):
            optimizer = get_optim(optimizer)

        self.optimizer = optimizer
        self.loss = loss

    @abstractmethod
    def train_step(self, im_features: ImageFeatures, target: Tensor) -> Dict:
        """"""

    @abstractmethod
    def eval_step(self, im_features: ImageFeatures, target: Tensor) -> Dict:
        """"""

    @abstractmethod
    def infer(self, im_features: ImageFeatures) -> Tensor:
        """"""
