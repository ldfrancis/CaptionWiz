import logging
import sys

from captionwiz.utils.config_utils import get_datetime

from .constants import DTIME, LOG_DIR, LOG_FILE, TENSORBOARD_LOG_DIR

# logging ##

# create logger
logger_name = "caption"
logger = logging.getLogger(logger_name)

# formatter
formatter = logging.Formatter(
    "%(module)s.py:%(lineno)d|%(levelname)s|%(name)s: %(message)s"
)

# file handler
LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
filehandler = logging.FileHandler(LOG_FILE)
filehandler.setFormatter(formatter)

# stream handler
streamhandlerStdOut = logging.StreamHandler(sys.stdout)
streamhandlerStdErr = logging.StreamHandler(sys.stderr)
streamhandlerStdOut.setFormatter(formatter)
streamhandlerStdErr.setFormatter(formatter)

# add handlers
logger.addHandler(filehandler)
logger.addHandler(streamhandlerStdOut)
logger.setLevel("DEBUG")

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
train_log_dir = tensorboard_log_dir / f"{dtime}/train"
test_log_dir = tensorboard_log_dir / f"{dtime}/test"

# define summary writers
train_log_dir.mkdir(parents=True, exist_ok=True)
test_log_dir.mkdir(parents=True, exist_ok=True)


def create_logger(name):
    log_file = LOG_DIR / f"{DTIME}_{name}.log"
    filehandler = logging.FileHandler(log_file)
    filehandler.setFormatter(formatter)
    logger = logging.getLogger(name)
    logger.addHandler(filehandler)
    logger.addHandler(streamhandlerStdOut)
    return logger
