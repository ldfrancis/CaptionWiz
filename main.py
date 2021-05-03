import click

from captionwiz.dataset import dataset_class
from captionwiz.model import model_class
from captionwiz.model.extractor import FeatureExtractor
from captionwiz.training import Trainer
from captionwiz.training.loss import caption_xe_loss
from captionwiz.training.optim import get_optim
from captionwiz.training.schedule import CaptionLrSchedule
from captionwiz.utils.config_utils import load_yaml, setup_wandb
from captionwiz.utils.constants import BASE_DIR
from captionwiz.utils.log_utils import logger

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
    help="Decide whether to log graph to tensorboard.",
)
@click.option(
    "--evaluate",
    "-ev",
    show_default=True,
    is_flag=True,
    help="Decide whether to log graph to tensorboard.",
)
@click.option(
    "--test",
    "-te",
    show_default=True,
    is_flag=True,
    help="Decide whether to log graph to tensorboard.",
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
    "--optim",
    "-o",
    default=cfg["optim"],
    show_default=True,
    type=click.Choice(["adam"]),
    help="Optimizer to use.",
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
    optim,
):
    logger.info("Setting config")
    # override config with command line arguments/flags
    # boolean flags
    cfg["log_graphs"] = log_graphs or cfg["log_graphs"]
    cfg["train"] = train if train else cfg["train"]
    cfg["test"] = test if test else cfg["test"]
    cfg["evaluate"] = evaluate if evaluate else cfg["evaluate"]
    # other settings
    cfg["name"] = name
    cfg["dataset"] = dataset
    cfg["trainer_loglevel"] = trainer_loglevel
    cfg["max_to_keep"] = max_to_keep
    cfg["extractor"] = extractor
    cfg["batch_size"] = batch_size
    cfg["epochs"] = epochs
    cfg["learning_rate"] = learning_rate
    # model
    cfg["caption_model"] = caption_model
    cfg["embedding_dim"] = embedding_dim
    cfg["units"] = units
    # trainer
    cfg["lr_decay"] = lr_decay
    cfg["lr_patience"] = lr_patience
    cfg["max_length"] = max_length

    logger.info(f"Done setting config\n\tConfig: {cfg}")

    setup_wandb(cfg)

    # get model, dataset, and trainer ready
    logger.info("Creating model, dataset, and trainer")
    _dataset = dataset_class[dataset]()
    caption_model_cfg = {
        "embedding_dim": embedding_dim,
        "units": units,
        "vocab_size": top_k_words or _dataset.vocab_size,
        "tokenizer": _dataset.tokenizer,
        "max_length": max_length,
        "name": caption_model,
    }
    _caption_model = model_class[caption_model](**caption_model_cfg)
    _feature_extractor = FeatureExtractor(extractor, _dataset, batch_size)
    _learning_rate = (
        CaptionLrSchedule(_caption_model, lr_decay, learning_rate, lr_patience),
    )
    _optimizer = get_optim(optim, _learning_rate)

    trainer = Trainer(
        _caption_model, _dataset, _feature_extractor, _optimizer, caption_xe_loss, cfg
    )

    # train:
    if cfg["train"]:
        """[summary]"""
        logger.info("Train")
        trainer.train(epochs)

    # eval:
    if cfg["eval"]:
        """[summary]"""
        logger.info("Eval")
        trainer.eval(True)

    # test
    if cfg["test"]:
        """[summary]"""
        logger.info("Test")
        trainer.test()
