import argparse
import pytorch_lightning as pl
from pytorch_lightning.utilities import rank_zero_info
import models
import tasks
import time
import torch
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


def test_model(args):  # might put on another file
    # python test_main.py --model_name TGCN --max_epochs 1 --batch_size 32 --loss mse_with_regularizer --settings supervised
    """Test trained model"""
    dm_test = utils.data.SpatioTemporalCSVDataModule(  # get data
        feat_path=DATA_PATHS["scada"]["train_clean_feat"],
        adj_path=DATA_PATHS["scada"]["adj_matrix"],
    )
    tgcn_model = get_model(dm_test)  # get model with input data
    task = get_task(tgcn_model, dm_test)  # supervised learning
    callbacks = get_callbacks()  # get callbacks, trained weights
    path = (
        "saved_models/time_21_41_24_date_12_30_2021_model.pt"  # load the trained weight
    )
    tgcn_model.load_state_dict(torch.load(path))

    # tgcn_model.eval()
    tester = pl.Trainer.from_argparse_args(args, callbacks=callbacks)
    tester.fit(task, dm_test)  # train model
    check_points = "lightning_logs/version_5/checkpoints/epoch=0-step=109.ckpt"

    testing_result = tester.test(
        ckpt_path=check_points, datamodule=dm_test
    )  # validate model

    return testing_result


def main(args):
    """User Interface"""
    rank_zero_info(vars(args))
    results = test_model(args)
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
