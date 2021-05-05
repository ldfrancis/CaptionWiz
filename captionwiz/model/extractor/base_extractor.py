import tensorflow as tf
from tensorflow.keras.applications import InceptionV3
from tensorflow.keras.applications.inception_v3 import preprocess_input


def get_inceptionV3():
    base_inceptionV3 = InceptionV3(include_top=False, weights="imagenet")

    def _extractor(img):
        img = preprocess_input(img)
        feat = base_inceptionV3(img)
        feat = tf.reshape(feat, shape=(feat.shape[0], -1, feat.shape[3]))
        feat = tf.squeeze(feat)
        return (feat,)

    return _extractor
