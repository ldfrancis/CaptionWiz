import collections
from abc import abstractmethod
from typing import Any, Dict, List, Tuple

from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer

from captionwiz.utils.type import FilePath


class CaptionDS:
    """A dataset for image captioning

    Attribuites:
    + train_im_to_caption: A dictionary that maps images to captions
    + train_image_caption_pairs: a list of image-caption pairs
    + val_im_to_caption: A dictionary that maps images to captions
    + val_image_caption_pairs: a list of image-caption pairs
    + test_images: a list of images (paths)
    """

    def __init__(self, name) -> None:
        """Initialize the attributes and call abstract method that sets im_to_caption
        and image_caption_pairs
        """
        self.name = name
        self.im_to_caption: Dict[Any, List[str]] = collections.defaultdict(list)
        self.image_caption_pairs: List[Tuple[FilePath, str]] = []
        self.create_image_caption_pairs()
        self.preprocess()
        self.vocab_size = len(self.tokenizer.word_index) + 1

    @abstractmethod
    def create_image_caption_pairs(self):
        """Method must set the im_to_caption and image_caption_pairs attributes"""

    def preprocess(self):

        # fit tokenizer to train split
        captions = [cap for _, cap in self.train_image_caption_pairs]
        self.train_img_paths = [im for im, _ in self.train_image_caption_pairs]
        tokenizer = Tokenizer(
            num_words=None, oov_token="<unk>", filters='!"#$%&()*+.,-/:;=?@[]^_`{|}~ '
        )
        tokenizer.fit_on_texts(captions)
        tokenizer.word_index["<pad>"] = 0
        tokenizer.index_word[0] = "<pad>"
        caption_seqs = tokenizer.texts_to_sequences(captions)
        self.train_caption_tensor = pad_sequences(caption_seqs, padding="post")
        self.max_length = max(len(s) for s in caption_seqs)
        self.tokenizer = tokenizer
        self.train_image_caption_pairs = [
            (im, cap)
            for im, cap in zip(self.train_img_paths, self.train_caption_tensor)
        ]

        # use tokenizer on val split
        captions = [cap for _, cap in self.val_image_caption_pairs]
        self.val_img_paths = [im for im, _ in self.val_image_caption_pairs]
        caption_seqs = tokenizer.texts_to_sequences(captions)
        self.val_caption_tensor = pad_sequences(
            caption_seqs, padding="post", maxlen=self.max_length
        )
        self.val_image_caption_pairs = [
            (im, cap) for im, cap in zip(self.val_img_paths, self.val_caption_tensor)
        ]

        # use also on test split
        dummy_cap = captions[0]
        self.test_img_paths = [im for im in self.test_images]
        captions = [dummy_cap] * len(self.test_img_paths)
        caption_seqs = tokenizer.texts_to_sequences(captions)
        self.test_caption_tensor = pad_sequences(
            caption_seqs, padding="post", maxlen=self.max_length
        )
        self.test_image_caption_pairs = [
            (im, cap) for im, cap in zip(self.test_img_paths, self.test_caption_tensor)
        ]
