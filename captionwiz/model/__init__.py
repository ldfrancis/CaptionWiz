from captionwiz.utils.constants import SHOW_ATT_TELL

from .captioning_model.caption_model import CaptionModel
from .captioning_model.show_attnd_tell import ShowAttTell

model_class = {SHOW_ATT_TELL: ShowAttTell}

__all__ = ["CaptionModel", "ShowAttTell"]
