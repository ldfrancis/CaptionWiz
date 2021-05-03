from captionwiz.utils.constants import MSCOCO, MSCOCO_KARPATHY

from .caption_ds import CaptionDS
from .mscoco import MSCOCO_CaptionDS
from .mscoco.mscoco_ds import MSCOCO_Karpathy_CaptionDS

dataset_class = {MSCOCO: MSCOCO_CaptionDS, MSCOCO_KARPATHY: MSCOCO_Karpathy_CaptionDS}

__all__ = ["CaptionDS", "MSCOCO_CaptionDS", "MSCOCO_Karpathy_CaptionDS"]
