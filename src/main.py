import argparse
import pytorch_lightning as pl
from pytorch_lightning.utilities import rank_zero_info
import models
import tasks
import time
import datetime
import utils.callbacks
import utils.data
import utils.logging

import os  # fix OMP error

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

DATA_PATHS = {
    "scada": {
        "train_clean_feat": "data/processed/processed_clean_scada_dataset.csv",
        "adj_matrix": "data/processed/processed_scada_adj_matrix.csv",
        "train_poison_feat": "data/processed/",
        "test_feat": "data/processed",
    },
}


def get_model(dm):
    """Return TGCN model"""
    model = models.TGCN(adj=dm.adj, hidden_dim=args.hidden_dim)
    return model


def get_task(model, dm):
    """Set Supervised Learning"""
    task = getattr(tasks, "SupervisedForecastTask")(
        model=model, feat_max_val=dm.feat_max_val
    )
    return task


def get_callbacks():
    checkpoint_callback = pl.callbacks.ModelCheckpoint(monitor="train_loss")
    plot_validation_predictions_callback = (
        utils.callbacks.PlotValidationPredictionsCallback(monitor="train_loss")
    )
    callbacks = [
        checkpoint_callback,
        plot_validation_predictions_callback,
    ]
    return callbacks


def main_supervised(args):
    """Main training setup for supervised learning with the parsed arguments"""
    dm = utils.data.SpatioTemporalCSVDataModule(  # get data
        feat_path=DATA_PATHS["scada"]["train_clean_feat"],
        adj_path=DATA_PATHS["scada"]["adj_matrix"],
    )
    model = get_model(dm)  # get model
    task = get_task(model, dm)  # supervised learning
    callbacks = get_callbacks()  # get callbacks, trained weights
    trainer = pl.Trainer.from_argparse_args(args, callbacks=callbacks)
    trainer.fit(task, dm)  # train model
    results = trainer.validate(datamodule=dm)  # validate model

    # save model
    # log training date
    x = datetime.datetime.now()
    x
    localtime = (
        "time_"
        + str(x.hour)
        + "_"
        + str(x.minute)
        + "_"
        + str(x.second)
        + "_date_"
        + str(x.month)
        + "_"
        + str(x.day)
        + "_"
        + str(x.year)
    )

    trainer.save_checkpoint("saved_models/" + localtime + "_model.ckpt")

    return results


def main(args):
    """User Interface"""
    rank_zero_info(vars(args))
    results = main_supervised(args)
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser = pl.Trainer.add_argparse_args(parser)

    parser.add_argument(
        "--data",
        type=str,
        help="The name of the dataset",
        choices=("scada"),
        default="scada",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        help="The name of the model for spatiotemporal prediction",
        choices=("TGCN"),
        default="TGCN",
    )
    parser.add_argument(
        "--settings",
        type=str,
        help="The type of settings, e.g. supervised learning",
        choices=("supervised",),
        default="supervised",
    )
    parser.add_argument(
        "--log_path", type=str, default=None, help="Path to the output console log file"
    )

    temp_args, _ = parser.parse_known_args()  # get variable

    parser = getattr(
        utils.data, temp_args.settings.capitalize() + "DataModule"
    ).add_data_specific_arguments(parser)
    parser = getattr(models, temp_args.model_name).add_model_specific_arguments(parser)
    parser = getattr(
        tasks, temp_args.settings.capitalize() + "ForecastTask"
    ).add_task_specific_arguments(parser)

    args = parser.parse_args()
    utils.logging.format_logger(pl._logger)

    results = main(args)
