import argparse
import logging
import os
import pprint
import sys
import time
from typing import TypeVar

import numpy as np
import torch

from h264.src.dataload.dataloader_csv import create_dataset, set_dataloader
from h264.src.models.h264_model import ModelSummary, compute_loss, create_model
from h264.src.stages.utils import (
    create_batch_stats,
    create_epoch_summary,
    jit_save_model,
    perf_eval_epoch,
    set_random_seed,
    setup_config,
    TrainStage,
)
from h264.src.visualization.train_curve import plot_train_curve


# Data type alias.
TDataLoader = TypeVar("TDataLoader", bound="torch.utils.data.dataloader.DataLoader")

# Set the logging for this application.
LOG = logging.getLogger(os.path.basename(__file__))


def collect_inputs_and_compute_loss(
    modelData: ModelSummary, loadedData: dict, flagEval: bool, config: dict
) -> float:
    """_summary_

    Args:
        modelData (ModelSummary): _description_
        loadedData (dict): _description_
        flagEval (bool): _description_
        config (dict): _description_

    Returns:
        float: _description_
    """
    loss = compute_loss(modelData, loadedData, config)

    if not flagEval:
        # Gradient decent
        modelData.optimizer.zero_grad(set_to_none=True)

        # Run backward step
        modelData.scaler.scale(loss).backward()

        # Optimizer gradient clipper.
        torch.nn.utils.clip_grad_norm_(
            modelData.model.parameters(),
            max_norm=config["train"]["clip_gradnorm_max"],
        )

        # Run optimization step
        modelData.scaler.step(modelData.optimizer)
        modelData.scaler.update()

    return loss


def get_traininfo(
    config: dict,
    trainLoader: TDataLoader,
    valLoader: TDataLoader,
) -> dict:
    """
    Prepare or load train info.

    Args:
        config (dict): the global parameters configuration.
        trainLoader:
        valLoader:

    Returns:
        data (dict): the training information initialized.
    """
    data = {
        TrainStage.TRAIN.name: [],
        TrainStage.VALIDATE.name: [],
        "timestamp": [],
        "last_epoch": 0,
        "best_epoch": 0,
        "best_loss": float("inf"),
        "num_train_batches": len(trainLoader),
        "num_val_batches": len(valLoader),
        "num_epochs": config["train"]["num_epochs"],
        "batch_size": config["train"]["batch_size"],
    }

    LOG.debug(pprint.pformat(data, indent=4))
    return data


def get_lr(modelData: ModelSummary) -> float:
    """Get the current learning rate from the optimizer."""
    lr = 0.0
    for paramGroup in modelData.optimizer.param_groups:
        lr = paramGroup["lr"]
    return lr


def train_model(
    modelData: ModelSummary,
    trainLoader: TDataLoader,
    valLoader: TDataLoader,
    config: dict,
) -> None:
    currentLearningRate = get_lr(modelData)
    trainInfo = get_traininfo(config, trainLoader, valLoader)

    # Main training-validation loop.
    for epochIdx in range(trainInfo["last_epoch"] + 1, trainInfo["num_epochs"]):
        ### 1) Start with the training iterations.
        timeStart = time.time()
        trainInfo["timestamp"].append(timeStart)
        runningStats = np.zeros(trainInfo["num_train_batches"])

        # modelData.model.set_mode("train")
        modelData.model.train()
        loadedData = None
        trainLoaderIter = iter(trainLoader)
        for batchIdx, loadedData in enumerate(trainLoaderIter):
            # Compute one model iteration.
            lossValue = collect_inputs_and_compute_loss(
                modelData, loadedData, False, config
            )
            runningStats[batchIdx] = lossValue.item()

            # Progress message.
            if batchIdx > 0 and batchIdx % config["train"]["show_epochs"] == 0:
                LOG.info(
                    create_batch_stats(runningStats, trainInfo, epochIdx, batchIdx)
                )

        # Create a summary for the training epoch.
        trainInfo[TrainStage.TRAIN.name].append(
            create_epoch_summary(
                runningStats,
                epochIdx,
                time.time() - timeStart,
                currentLearningRate,
            )
        )

        ### 2) At the end of each epoch, obtain the validation loss.
        timeStart = time.time()
        runningStats = np.zeros(trainInfo["num_val_batches"])

        # modelData.model.set_mode("eval")
        modelData.model.eval()
        valLoaderIter = iter(valLoader)
        for batchIdx, loadedData in enumerate(valLoaderIter):
            with torch.no_grad():
                lossValue = collect_inputs_and_compute_loss(
                    modelData, loadedData, True, config
                )
                runningStats[batchIdx] = lossValue.item()

        trainInfo[TrainStage.VALIDATE.name].append(
            create_epoch_summary(
                runningStats,
                epochIdx,
                time.time() - timeStart,
                currentLearningRate,
            )
        )

        ### 3) Adjust the learning rates based on losses.
        bestModelFlag = False
        if config["train"]["scheduler"] == "cosine":
            modelData.optimizer = modelData.scheduler.update_lr(
                optimizer=modelData.optimizer, epoch=epochIdx, curr_iter=epochIdx
            )
        else:
            modelData.scheduler.step(trainInfo[TrainStage.VALIDATE.name][-1]["loss"])
        if trainInfo[TrainStage.VALIDATE.name][-1]["loss"] <= trainInfo["best_loss"]:
            trainInfo["best_loss"] = trainInfo[TrainStage.VALIDATE.name][-1]["loss"]
            trainInfo["best_epoch"] = epochIdx
            bestModelFlag = True

        ### 4) Update the current learning rate.
        currentLearningRate = get_lr(modelData)

        ### 5) Print the epoch results, save the best model and render the training curves
        LOG.info(perf_eval_epoch(trainInfo, config["train"]["model_path"], TrainStage.TRAIN, epochIdx))
        LOG.info(perf_eval_epoch(trainInfo, config["train"]["model_path"], TrainStage.VALIDATE, epochIdx))
        if bestModelFlag and epochIdx >= 3:
            jitFilename = config["train"]["model_path"] / "best_model.pt"
            LOG.info(
                "Saving the current best model from epoch %d to %s.",
                epochIdx,
                jitFilename,
            )
            if loadedData:
                jit_save_model(
                    jitFilename,
                    modelData.model,
                    loadedData["h264"].to(config["model"]["device"], non_blocking=True),
                )

        ### 6) Plot train curve and overwrite the latest model.
        if epochIdx > 1:
            LOG.info(
                "Epoch %d - Best epoch: %d, loss: %f.",
                epochIdx,
                trainInfo["best_epoch"],
                trainInfo["best_loss"],
            )
            plot_train_curve(config, trainInfo)
            modelFilename = config["train"]["model_path"] / "last_model.pt"
            if loadedData:
                jit_save_model(
                    modelFilename,
                    modelData.model,
                    loadedData["h264"].to(config["model"]["device"], non_blocking=True),
                )


def setup_and_train(configPath: str, baseDir: str = ".") -> None:
    """
    Setup before calling the training function.

    Args:
        configPath (str): the path to the yaml configuration file.
        baseDir (str): the reference directory for the configuration links.
    """
    config = setup_config(configPath, baseDir)
    set_random_seed(config["base"]["random_seed"])

    # Set the logger configuration.
    logging.basicConfig(
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        level=logging.getLevelName(config["base"]["logging"]),
        stream=sys.stdout,
    )

    # Create a dataset.
    dataset = create_dataset(config)

    # Get the dataset loaders.
    trainLoader, valLoader, _ = set_dataloader(config, dataset)
    LOG.info(
        "Training set > samples: %d, batches: %d, batch_size: %d",
        len(trainLoader.dataset),
        len(trainLoader),
        trainLoader.batch_size,
    )
    LOG.info(
        "Validation set > samples: %d, batches: %d, batch_size: %d",
        len(valLoader.dataset),
        len(valLoader),
        valLoader.batch_size,
    )

    # Create a model and support optimization elements.
    modelData = create_model(config)

    # Run the training phases.
    LOG.info("Starting the model training process.")
    train_model(modelData, trainLoader, valLoader, config)
    LOG.info("Completed the model training process.")


if __name__ == "__main__":
    argsParser = argparse.ArgumentParser()
    argsParser.add_argument("--config", dest="config", required=True)
    args = argsParser.parse_args()

    setup_and_train(configPath=args.config)
