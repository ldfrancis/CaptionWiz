import logging

import tensorflow as tf

from captionwiz.utils.config_utils import get_datetime

from .constants import LOG_FILE, TENSORBOARD_LOG_DIR

# logging ##

# create logger
logger_name = "caption"
logger = logging.getLogger(logger_name)

# formatter
formatter = logging.Formatter("%(asctime)s|%(levelname)s|%(name)s: %(message)s")

# file handler
filehandler = logging.FileHandler(LOG_FILE)
filehandler.setFormatter(formatter)

# stream handler
streamhandler = logging.StreamHandler()
streamhandler.setFormatter(formatter)

# add handlers
logger.addHandler(filehandler)
logger.addHandler(streamhandler)

# logging levels map
logging_levels = {
    "debug": logging.DEBUG,
    "info": logging.INFO,
    "warning": logging.WARNING,
    "error": logging.ERROR,
    "critical": logging.CRITICAL,
}


# tensorboard ##

# define log dir
tensorboard_log_dir = TENSORBOARD_LOG_DIR
dtime = get_datetime("%Y%m%d-%H%M%S")
train_log_dir = tensorboard_log_dir / "{dtime}/train"
test_log_dir = tensorboard_log_dir / "{dtime}/test"

# define summary writers
train_summary_writer = tf.summary.create_file_writer(train_log_dir)
test_summary_writer = tf.summary.create_file_writer(test_log_dir)
