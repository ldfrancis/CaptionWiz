import click
import tensorflow as tf
import yaml

from captionwiz.dataset import dataset_class
from captionwiz.model import model_class
from captionwiz.training import Trainer
from captionwiz.training.loss import caption_xe_loss
from captionwiz.training.optim import get_optim
from captionwiz.utils.config_utils import load_yaml, setup_wandb
from captionwiz.utils.constants import BASE_DIR
from captionwiz.utils.log_utils import logger, test_log_dir, train_log_dir

cfg = load_yaml(BASE_DIR / "config.yaml")


@click.command()
@click.option(
    "--name",
    "-n",
    default=cfg["name"],
    show_default=True,
    type=str,
    help="Name of experiment.",
)
@click.option(
    "--dataset",
    "-d",
    default=cfg["dataset"],
    show_default=True,
    type=click.Choice(["mscoco", "vizwiz"]),
    help="Dataset to use.",
)
@click.option(
    "--caption_model",
    "-c",
    default=cfg["caption_model"],
    show_default=True,
    type=click.Choice(["show_att_tell"]),
    help="The caption model to use.",
)
@click.option(
    "--trainer_loglevel",
    "-tl",
    default=cfg["trainer_loglevel"],
    show_default=True,
    type=click.Choice(["debug", "info", "warning", "error", "critical"]),
    help="Log level for trainer.",
)
@click.option(
    "--max_to_keep",
    "-m",
    default=cfg["max_to_keep"],
    show_default=True,
    type=int,
    help="max checkpoints to save.",
)
@click.option(
    "--extractor",
    "-e",
    default=cfg["extractor"],
    show_default=True,
    type=click.Choice(["inceptionv3"]),
    help="Image feature extractor to use.",
)
@click.option(
    "--log_graphs",
    "-l",
    show_default=True,
    is_flag=True,
    help="Decide whether to log graph to tensorboard.",
)
@click.option(
    "--batch_size",
    "-b",
    default=cfg["batch_size"],
    show_default=True,
    type=int,
    help="Batch size to use for dataloader.",
)
@click.option(
    "--epochs",
    "-ep",
    default=cfg["epochs"],
    show_default=True,
    type=int,
    help="Number of epochs to train for.",
)
@click.option(
    "--learning_rate",
    "-lr",
    default=cfg["lr"],
    show_default=True,
    type=float,
    help="Initial learning rate",
)
@click.option(
    "--train",
    "-tr",
    show_default=True,
    is_flag=True,
    help="Decide whether to train on train split.",
)
@click.option(
    "--evaluate",
    "-ev",
    show_default=True,
    is_flag=True,
    help="Decide whether to evaluate with val split.",
)
@click.option(
    "--test",
    "-te",
    show_default=True,
    is_flag=True,
    help="Decide whether to test model using dataset test split.",
)
@click.option(
    "--analyze_data",
    "-ad",
    show_default=True,
    is_flag=True,
    help="Decide whether to analyze the train split of the dataset.",
)
@click.option(
    "--restore",
    "-r",
    show_default=True,
    is_flag=True,
    help="Decide whether to restore model from checkpoint.",
)
@click.option(
    "--embedding_dim",
    "-ed",
    default=cfg["embedding_dim"],
    show_default=True,
    type=int,
    help="Dimension for Image embedding.",
)
@click.option(
    "--units",
    "-u",
    default=cfg["units"],
    show_default=True,
    type=int,
    help="Number of hidden units to use.",
)
@click.option(
    "--epochs",
    "-ep",
    default=cfg["epochs"],
    show_default=True,
    type=int,
    help="Number of epochs to train for.",
)
@click.option(
    "--top_k_words",
    "-tw",
    default=cfg["top_k_words"],
    show_default=True,
    type=int,
    help="Number of top words to use for vocab_size.",
)
@click.option(
    "--max_length",
    "-ml",
    default=cfg["max_length"],
    show_default=True,
    type=int,
    help="Maximum sequence length.",
)
@click.option(
    "--lr_decay",
    "-ld",
    default=cfg["lr_decay"],
    show_default=True,
    type=float,
    help="learning rate decay multiplier.",
)
@click.option(
    "--lr_patience",
    "-lp",
    default=cfg["lr_patience"],
    show_default=True,
    type=int,
    help="Steps to wait for before decaying lr.",
)
@click.option(
    "--word_count_threshold",
    "-wct",
    default=cfg["word_count_threshold"],
    show_default=True,
    type=int,
    help="Threshold used to decide which words to keep in vocab.",
)
@click.option(
    "--optimizer",
    "-o",
    default=cfg["optimizer"],
    show_default=True,
    type=click.Choice(["adam"]),
    help="Optimizer to use.",
)
@click.option(
    "--shuffle",
    "-sh",
    show_default=True,
    is_flag=True,
    help="Whether to shuffle data when creating dataloaders.",
)
def main(
    name,
    dataset,
    caption_model,
    trainer_loglevel,
    max_to_keep,
    extractor,
    log_graphs,
    batch_size,
    epochs,
    learning_rate,
    train,
    evaluate,
    test,
    embedding_dim,
    units,
    top_k_words,
    max_length,
    lr_decay,
    lr_patience,
    optimizer,
    shuffle,
    analyze_data,
    word_count_threshold,
    restore,
):
    logger.info("Setting config")

    # override config with command line arguments/flags

    # boolean flags
    cfg["log_graphs"] = log_graphs
    cfg["train"] = train
    cfg["test"] = test
    cfg["evaluate"] = evaluate
    cfg["shuffle"] = shuffle
    cfg["analyze_data"] = analyze_data
    cfg["restore"] = restore

    # other settings
    cfg["name"] = name
    cfg["dataset"] = dataset
    cfg["trainer_loglevel"] = trainer_loglevel
    cfg["max_to_keep"] = max_to_keep
    cfg["extractor"] = extractor
    cfg["batch_size"] = batch_size
    cfg["epochs"] = epochs
    cfg["learning_rate"] = learning_rate
    cfg["word_count_threshold"] = word_count_threshold
    cfg["top_k_words"] = top_k_words

    # model
    cfg["caption_model"] = caption_model
    cfg["embedding_dim"] = embedding_dim
    cfg["units"] = units

    # trainer
    cfg["lr_decay"] = lr_decay
    cfg["lr_patience"] = lr_patience
    cfg["max_length"] = max_length
    cfg["restart"] = float(cfg["restart"])

    pretty_cfg = yaml.dump(cfg, sort_keys=False, default_flow_style=False)

    logger.info(f"Done setting config\n\nConfig: \n{pretty_cfg}")

    setup_wandb(cfg)
    train_summary_writer = tf.summary.create_file_writer(str(train_log_dir.absolute()))
    test_summary_writer = tf.summary.create_file_writer(str(test_log_dir.absolute()))
    cfg["train_summary_writer"] = train_summary_writer
    cfg["test_summary_writer"] = test_summary_writer

    # get model, dataset, and trainer ready
    logger.info("Creating model, dataset, and trainer")
    _dataset = dataset_class[dataset](cfg)

    # analyze data:
    if cfg["analyze_data"]:
        """Analyze the train split"""
        logger.info(f"Analyzing the train split of dataset {dataset}")
        _dataset.analyze(word_count_threshold)
        return

    caption_model_cfg = {
        "embedding_dim": embedding_dim,
        "units": units,
        "vocab_size": top_k_words or _dataset.vocab_size,
        "tokenizer": _dataset.tokenizer,
        "max_length": max_length or _dataset.max_length,
    }

    _caption_model = model_class[caption_model](**caption_model_cfg)
    # _learning_rate = CaptionLrSchedule(
    #     _caption_model, lr_decay, learning_rate, lr_patience
    # )
    _optimizer = get_optim(optimizer, learning_rate)

    trainer = Trainer(_caption_model, _dataset, _optimizer, caption_xe_loss, cfg)

    # train:
    if cfg["train"]:
        """Train the caption model"""
        logger.info("Train")
        trainer.train(epochs)

    # evaluate:
    if cfg["evaluate"]:
        """Perform evaluation on the val split using the caption model"""
        logger.info("Eval")
        trainer.eval(True)

    # test
    if cfg["test"]:
        """Inference on the test split using the caption model"""
        logger.info("Test")
        trainer.test()
