import logging
from time import time
from typing import Dict, List

import numpy as np
import tensorflow as tf
from tensorflow.keras.optimizers import Optimizer
from tqdm import tqdm

from captionwiz.dataset import CaptionDS
from captionwiz.model import CaptionModel
from captionwiz.model.extractor import FeatureExtractor
from captionwiz.utils.config_utils import get_datetime
from captionwiz.utils.constants import CHECKPOINT_DIR, LOG_DIR
from captionwiz.utils.log_utils import (
    formatter,
    logger,
    logging_levels,
    streamhandlerStdOut,
)
from captionwiz.utils.type import ImageFeatures, Loss, Tensor


class Trainer:
    """Trains a caption model given a dataset, and an optimizer and loss"""

    def __init__(
        self,
        caption_model: CaptionModel,
        dataset: CaptionDS,
        optimizer: Optimizer,
        loss: Loss,
        cfg: Dict,
        name: str = None,
    ):
        """Initialize the trainer

        Args:
            caption_model (CaptionModel): The image captioning model to be trained
            dataset (CaptionDS): Caption dataset to use for training
            optimizer (Optimizer): Optimizer to use. usually adam
            loss (Loss): Loss function. usually cross entropy
            cfg (Dict): Program
            name (str, optional): Name of the trainer. Defaults to None.
        """
        self._caption_model = caption_model
        self._caption_model.optim_prep(optimizer, loss)
        self._dataset = dataset
        self._optimizer = self._caption_model.optimizer
        self._loss = self._caption_model.loss
        self._cfg = cfg
        self._batch_size = cfg["batch_size"]
        self._buffer_size = cfg["buffer_size"]
        self._shuffle = cfg["shuffle"]

        self.current_epoch = tf.Variable(0)
        self.current_step = tf.Variable(0)

        self.name = name or f"{caption_model.name}_{dataset.name}"

        self.best_loss = tf.Variable(np.inf, dtype=tf.float32)

        self.create_checkpoint()
        self.create_logger()

        extractor_name = self._cfg["extractor"]
        self._extractor = FeatureExtractor(
            extractor_name, self._dataset, self._batch_size
        )

        self._train_dataloader = self._extractor.create_image_caption_dataloader(
            buffer_size=self._buffer_size,
            batch_size=self._batch_size,
            shuffle=self._shuffle,
            split="train",
        )
        self._eval_dataloader = self._extractor.create_image_caption_dataloader(
            buffer_size=self._buffer_size,
            batch_size=self._batch_size,
            shuffle=self._shuffle,
            split="val",
        )
        self._test_dataloader = self._extractor.create_image_caption_dataloader(
            buffer_size=self._buffer_size,
            batch_size=self._batch_size,
            shuffle=False,
            split="test",
        )

        self.test_summary_writer = self._cfg["test_summary_writer"]
        self.train_summary_writer = self._cfg["train_summary_writer"]

        logger.info(
            f"Trainer, {self.name}, created:-\n"
            f"checkpoint directory: {self.checkpoint_path}\n"
            f"log file: {self.log_file}"
        )

    def create_checkpoint(self):
        """Checkpoint for saving train progress and enabling training resumption"""
        max_to_keep = self._cfg["max_to_keep"]
        self.checkpoint_path = CHECKPOINT_DIR / self.name
        self.ckpt = tf.train.Checkpoint(
            caption_model=self._caption_model,
            optimizer=self._optimizer,
            step=self.current_step,
            best_loss=self.best_loss,
        )
        self.ckpt_manager = tf.train.CheckpointManager(
            self.ckpt, self.checkpoint_path, max_to_keep=max_to_keep
        )

    def create_logger(self):
        """Create logger for longing train info"""
        dtime = get_datetime()
        self.log_file = LOG_DIR / f"{self.name}_{dtime}.log"

        log_level = self._cfg["trainer_loglevel"]

        filehandler = logging.FileHandler(self.log_file)
        filehandler.setFormatter(formatter)

        self.logger = logging.getLogger(self.name)
        self.logger.setLevel(logging_levels[log_level])
        self.logger.addHandler(filehandler)
        self.logger.addHandler(streamhandlerStdOut)

        self.logger.info(f"Created logger for trainer {self.name}")

    def summarise_metric(self, name: str, values: List[float]) -> Dict[str, float]:
        """Summarise a metric by computing its mean, max and min value
        given a list of such metric

        Args:
            name (str): Name of metric
            values (List[str]): List of metric values usually collected
            for each step over an epoch of training

        Returns:
            [Dict[str, float]]: summary
        """
        summarised_metric = {
            f"{name}_mean": np.mean(values),
            f"{name}_max": np.max(values),
            f"{name}_min": np.min(values),
        }
        return summarised_metric

    def eval(self, epoch, log=False):
        start_time = time()
        epoch_losses = []
        for batch, (img_features, target) in enumerate(self._eval_dataloader):
            # trace the eval_step func
            batch_loss = self.eval_step(img_features, target, batch, epoch)
            epoch_losses += [batch_loss.numpy()]
            self.current_step.assign_add(1)

        eval_losses = self.summarise_metric("eval_loss", epoch_losses)
        eval_time = time() - start_time

        if log:
            self.logger.info(
                f"\tval time: {eval_time}\n"
                f"\tval_loss_mean: {eval_losses['eval_loss_mean']}\n"
                f"\tval_loss_max: {eval_losses['eval_loss_max']}\n"
                f"\tval_loss_min: {eval_losses['eval_loss_min']}"
            )

            with self.test_summary_writer.as_default():
                for k, v in eval_losses.items():
                    tf.summary.scalar(k, v, step=self.current_step.numpy())
                tf.summary.scalar(
                    "val_loss",
                    eval_losses["eval_loss_mean"],
                    step=self.current_step.numpy(),
                )

        return eval_losses, eval_time

    def eval_step(
        self, img_features: ImageFeatures, target: Tensor, batchi: int, epoch: int
    ):
        if batchi == 0 and self._cfg["log_graphs"]:
            tf.summary.trace_on(graph=True, profiler=True)
            batch_loss, _ = self._caption_model.eval_step(img_features, target)
            with self.test_summary_writer.as_default():
                tf.summary.trace_export(name="eval_step_trace", step=epoch)
        else:
            batch_loss, _ = self._caption_model.eval_step(img_features, target)

        return batch_loss

    def train_step(
        self, img_features: ImageFeatures, target: Tensor, batchi: int, epoch: int
    ):
        if batchi == 0 and self._cfg["log_graphs"]:
            tf.summary.trace_on(graph=True, profiler=True)
            batch_loss, _ = self._caption_model.train_step(img_features, target)
            with self.train_summary_writer.as_default():
                tf.summary.trace_export(
                    name="train_step_trace", step=self.current_step.numpy()
                )
        else:
            batch_loss, _ = self._caption_model.train_step(img_features, target)

        return batch_loss

    def train_one_epoch(self, epoch):
        start_time = time()
        epoch_losses = []
        for batch, (img_features, target) in enumerate(self._train_dataloader):
            # trace the train_step func
            batch_loss = self.train_step(img_features, target, batch, epoch)
            epoch_losses += [batch_loss.numpy()]
            self.current_step.assign_add(1)
            with self.train_summary_writer.as_default():
                tf.summary.scalar(
                    "train_loss", batch_loss, step=self.current_step.numpy()
                )
                tf.summary.scalar(
                    "learning_rate",
                    self._optimizer.learning_rate._lr,
                    step=self.current_step.numpy(),
                )

        train_losses = self.summarise_metric("train_loss", epoch_losses)
        epoch_train_time = time() - start_time

        self.current_epoch.assign_add(1)

        return train_losses, epoch_train_time

    def train(self, epochs=10):
        """ """
        if self.ckpt_manager.latest_checkpoint and self._cfg["restore"]:
            self.ckpt.restore(self.ckpt_manager.latest_checkpoint)

            self.logger.info(
                f"Restored checkpoint from {self.checkpoint_path}"
                f"/{self.ckpt_manager.latest_checkpoint}"
                f" at epoch {self.current_epoch.numpy()} and step "
                f"{self.current_step.numpy()}"
            )

        self.logger.info("Commenced training")
        for epoch in range(self.current_epoch.numpy(), epochs + 1):
            self.logger.info(f"EPOCH {epoch+1}")
            train_losses, epoch_train_time = self.train_one_epoch(epoch)
            eval_losses, eval_time = self.eval(epoch)
            self.logger.info(
                f"Epoch {epoch}:- \n"
                f"\tepoch train time: {epoch_train_time}\n"
                f"\ttrain_loss_mean: {train_losses['train_loss_mean']}\n"
                f"\ttrain_loss_max: {train_losses['train_loss_max']}\n"
                f"\ttrain_loss_min: {train_losses['train_loss_min']}\n"
                f"\tval time: {eval_time}\n"
                f"\tval_loss_mean: {eval_losses['eval_loss_mean']}\n"
                f"\tval_loss_max: {eval_losses['eval_loss_max']}\n"
                f"\tval_loss_min: {eval_losses['eval_loss_min']}"
            )

            if tf.greater(self.best_loss, eval_losses["eval_loss_mean"]):
                self.best_loss.assign(eval_losses["eval_loss_mean"])
                save_path = self.ckpt_manager.save()
                self.logger.info(
                    f"Saving checkpoint with loss "
                    f"{eval_losses['eval_loss_mean']:1.2f} at {save_path}"
                )

            with self.train_summary_writer.as_default():
                for k, v in train_losses.items():
                    tf.summary.scalar(k, v, step=self.current_step.numpy())

            with self.test_summary_writer.as_default():
                for k, v in eval_losses.items():
                    tf.summary.scalar(k, v, step=self.current_step.numpy())

            self.ckpt.step.assign_add(1)

    def test(self):
        if self.ckpt_manager.latest_checkpoint and not self._cfg["train"]:
            self.ckpt.restore(self.ckpt_manager.latest_checkpoint)
            self.logger.info(
                f"Restored checkpoint from {self.checkpoint_path}"
                f"/{self.ckpt_manager.latest_checkpoint}"
                f" at epoch {self.current_epoch.numpy()} and step "
                f"{self.current_step.numpy()}"
            )

        imgs = []
        captions = []

        for (img_features, _) in tqdm(self._test_dataloader):
            captions_ = self._caption_model.infer(img_features)
            captions += [captions_.numpy()]
            imgs += [img_features.numpy()]

        imgs = np.concatenate(imgs, axis=0)
        captions = np.concatenate(captions, axis=0)
        texts = self._dataset.tokenizer.sequences_to_texts(captions)
        return captions, texts
