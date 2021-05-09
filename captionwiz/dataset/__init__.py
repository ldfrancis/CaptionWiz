from captionwiz.utils.constants import MSCOCO, MSCOCO_KARPATHY, VIZWIZ

from .caption_ds import CaptionDS
from .mscoco import MSCOCO_CaptionDS
from .mscoco.mscoco_ds import MSCOCO_Karpathy_CaptionDS
from .vizwiz.vizwiz_ds import VIZWIZ_CaptionDS

dataset_class = {
    MSCOCO: MSCOCO_CaptionDS,
    MSCOCO_KARPATHY: MSCOCO_Karpathy_CaptionDS,
    VIZWIZ: VIZWIZ_CaptionDS,
}

__all__ = [
    "CaptionDS",
    "MSCOCO_CaptionDS",
    "MSCOCO_Karpathy_CaptionDS",
    "VIZWIZ_CaptionDS",
]
