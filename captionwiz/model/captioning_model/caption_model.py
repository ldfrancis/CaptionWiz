from abc import abstractmethod
from typing import Dict, Union

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
        name="ShowAttTell",
    ):
        super(CaptionModel, self).__init__(name=name)
        self.tokenizer = tokenizer
        self.name = name

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
