from tensorflow.keras.optimizers import Adam

from captionwiz.utils.constants import ADAM


def get_optim(name, lr):
    optim = optim_map[name](lr)
    return optim


optim_map = {ADAM: Adam}
