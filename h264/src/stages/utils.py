import copy
import datetime
import logging
import os
import re
import shutil
from enum import Enum, auto
from pathlib import Path
from typing import Any, Dict, Iterable, Tuple, TypeVar, Union

import numpy as np
import random
import torch
import yaml
from matplotlib import pyplot as plt
from torch.nn.utils.weight_norm import WeightNorm

# Data type alias.
TJitModel = TypeVar("TJitModel", bound="torch.jit._script.RecursiveScriptModule")


# Define the logger for this library.
LOG = logging.getLogger(__name__)
LOG.addHandler(logging.NullHandler())


class TrainStage(Enum):
    """Enumeration for model training stages."""

    TRAIN = auto()
    VALIDATE = auto()
    TEST = auto()


def setup_config(
    configPath: str = "params.yaml", baseFolder: str = ".", evaluate: bool = False
) -> dict:
    """
    Utility function for parsing the configuration file, while also overwriting
    parameters for special conditions.

    Note:
        The function will add elements not found in the input YAML file. Those
        are added because of the environment 'root_folder' check. The list
        includes: config["base"]["base_dir"], config["dataset"]["base_dir"], and
        config["train"]["model_path"].

    Args:
        configPath:
            The path to the yaml configuration file.

        baseFolder:
            The reference directory for the configuration links.

        evaluate:
            The flag for resuming training and skip the folder creation step.

    Returns
        config:
           Returns the global parameters configuration.

    """
    baseDir = Path(baseFolder)

    # Load the source configuration file to set globally accessible parameters.
    configPath = baseDir / configPath

    loader = yaml.SafeLoader
    loader.add_implicit_resolver(
        u'tag:yaml.org,2002:float',
        re.compile(u'''^(?:
         [-+]?(?:[0-9][0-9_]*)\\.[0-9_]*(?:[eE][-+]?[0-9]+)?
        |[-+]?(?:[0-9][0-9_]*)(?:[eE][-+]?[0-9]+)
        |\\.[0-9_]+(?:[eE][-+][0-9]+)?
        |[-+]?[0-9][0-9_]*(?::[0-5]?[0-9])+\\.[0-9_]*
        |[-+]?\\.(?:inf|Inf|INF)
        |\\.(?:nan|NaN|NAN))$''', re.X),
        list(u'-+0123456789.'))

    cfg = yaml.load(open(configPath, "r", encoding="utf8"), Loader=loader)

    # Check docker for setting the base directory.
    if os.environ.get("DOCKER_RUNNING") == "True":
        localData = Path("/data")
    else:
        localData = Path(str(os.environ.get("DOCKER_DATA_MNT")))

    dataDir = localData / cfg["dataset"]["data_dir"]

    # Create a folder for saving the data.
    trainFolder = dataDir / cfg["train"]["save_folder"]
    if not evaluate:
        datetimeCode = (
            str(datetime.datetime.now())[:-7].replace(" ", "-").replace(":", "-")
        )
        versionMsg = "-".join(cfg["dataset"]["versions"])
        newModelFolder = (
            f"{cfg['model']['select_name']}"
            f"_{cfg['dataset']['base_name']}"
            f"_{versionMsg}"
            f"_{cfg['base'].get('user', 'valois')}"
            f"_{datetimeCode}"
        )
        modelDir = trainFolder / newModelFolder

        # Create a folder where model snapshots will be saved.
        modelDir.mkdir(parents=True, exist_ok=True)

        # copy params.yaml to model path
        shutil.copy(configPath, modelDir / Path(configPath).name)
    else:
        modelDir = trainFolder / cfg["evaluate"]["model_folder"]

    # We update the locations in the global config.
    cfg["base"]["base_dir"] = baseDir
    cfg["dataset"]["base_dir"] = dataDir
    cfg["train"]["model_path"] = modelDir

    return cfg


def create_batch_stats(
    runningStats: Union[np.ndarray, float], trainInfo: dict, currentEpoch: int, currentBatch: int, runningStats2: Union[np.ndarray, float] = None
) -> str:
    """
    Performance evaluation for batch.

    Args:
        runningStats:
            A numpy array with the statistics at each batch or a float

        trainInfo:
            The record of parameters related to the batch.

        currentEpoch:
            The the training epoch number.

        currentBatch:
            The batch number.

    Returns:
        msg:
            Returns a formatted string with the batch statistics.
    """

    if type(runningStats) is np.ndarray:
        perf = np.sum(runningStats)
    else:
        perf = runningStats

    if runningStats2 is not None:
        if type(runningStats2) is np.ndarray:
            perf2 = np.sum(runningStats2)
        else:
            perf2 = runningStats2
    msg = (
        f"Epoch [{currentEpoch:04d}/{trainInfo['num_epochs']:04d}]"
        f", Batch [B{trainInfo['batch_size']:02d}"
        f" {currentBatch:04d}/{trainInfo['num_train_batches']:04d}]"
        f", Perf [{perf / currentBatch:.5f}]"
    )

    if runningStats2 is not None:
        if type(runningStats2) is np.ndarray:
            perf2 = np.sum(runningStats2)
        else:
            perf2 = runningStats2
        msg += f", Perf2 [{perf2 / currentBatch:.5f}]"

    return msg


def create_epoch_summary(
    runningStats: Union[np.ndarray, float],
    currentEpoch: int,
    elapsedTime: float,
    learningRate: float,
    runningStats2: Union[np.ndarray, float] = None,
) -> dict:
    """
    Create a dictionary of metrics for the epoch.

    Args:
        runningStats:
            A numpy array with the statistics at each batch or a float

        currentEpoch:
            The training epoch number.

        elapsedTime:
            The time taken to run the epoch.

        learningRate:
            The learning rate used by the optimizer.

    Returns:
        summary:
            Returns the reformatted data sources.
    """

    if type(runningStats) is np.ndarray:
        loss = np.mean(runningStats)
    else:
        loss = runningStats
        
    
    summary = {
        "loss": loss,
        "epoch": currentEpoch,
        "time": elapsedTime,
        "lr": learningRate,
    }

    if runningStats2 is not None:
        if type(runningStats2) is np.ndarray:
            perf2 = np.mean(runningStats2)
        else:
            perf2 = runningStats2
    
        summary["loss2"] = perf2
    return summary

def perf_eval_epoch(
    trainInfo: dict, model_filepath: str, stage: TrainStage, currentEpoch: int
) -> str:
    """
    Performance evaluation for epoch, while also saving the information to file.

    Args:
        trainInfo:
            The record of parameters related to the batch.

        model_filepath:
            The model File path

        stage:
            The enumerated 'Train' or 'Valid' training step.

        currentEpoch:
            The epoch number.

    Returns:
        msg:
            Returns a formatted string with the epoch statistics.
    """
    msg = f'Epoch [{currentEpoch:04d}/{trainInfo["num_epochs"]:04d}], '
    msg += f"{stage.name} ["
    formats = {
        "epoch": "{k}: {v}",
        "time": "{k}: {v:.1f}s",
        "lr": "{k}: {v:.9f}",
    }
    defaultFormat = "{k}: {v:.5f}"
    msg += ", ".join(
        formats.get(k, defaultFormat).format(k=k, v=v)
        for k, v in trainInfo[stage.name][-1].items()
    )
    msg += "]"

    outputFilename = model_filepath / "perf_epoch.txt"
    with open(outputFilename, "a", encoding="utf8") as file:
        file.write(msg + "\n")

    return msg


def remove_weight_norm(module):
    module_list = [mod for mod in module.children()]
    if len(module_list) == 0:
        for k, hook in module._forward_pre_hooks.items():
            if isinstance(hook, WeightNorm):
                hook.remove(module)
                del module._forward_pre_hooks[k]
    else:
        for mod in module_list:
            remove_weight_norm(mod)


def jit_load_model(path: Path, device: str) -> TJitModel:
    """
    Utility function for load a PyTorch model using the JIT interface.

    Args:
        path (Path): the source model filename.

    Returns:
        (TModel): the loaded PyTorch model information.
    """
    LOG.info("Loading jit model [%s] on device [%s].", path, device)
    return torch.jit.load(path, map_location=device)


def jit_save_model(path: Path, model: TJitModel, modelInputs: torch.Tensor) -> None:
    """
    Utility function for saving a PyTorch model using the JIT interface

    Args:
        path:
            The destination filename for the model information.

        model:
            Wrapper for the PyTorch model.

        modelInputs:
            sample data to run the feedforward model.
    """
    model_copy = copy.deepcopy(model)
    remove_weight_norm(model_copy)
    traced = torch.jit.trace(model_copy, modelInputs, strict=False)
    torch.jit.save(traced, path)


def load_state(model_path: Path) -> dict:
    """
    Utility function for loading a pytorch model parameters.

    Fields in dict:
        'best_epoch', 'best_loss', 'model_state_dict',
        'optimizer_state_dict', 'scheduler_state_dict',
        'scaler_state_dict'
    Args:
        path (Path): the source file for the model data.
    """
    state_dict = torch.load(model_path, map_location="cpu")

    return state_dict


def save_state(epoch: int, best_epoch: int, model_data: dict, path: Path) -> None:
    """
    Utility function for saving the training state.

    Args:
        epoch (int): the epoch number.
        best_epoch (int): the epoch with the lowest validation loss value.
        model_data (dict): the model and optimization information.
        path (Path): the destination file.
    """
    torch.save(
        {
            "epoch": epoch,
            "best_epoch": best_epoch,
            "model_state_dict": model_data["model"].state_dict(),
            "optimizer_state_dict": model_data["optimizer"].state_dict(),
            "scheduler_state_dict": model_data["scheduler"].state_dict(),
            "scaler_state_dict": model_data["scaler"].state_dict(),
        },
        path,
        _use_new_zipfile_serialization=False,
    )


def set_random_seed(seed: int) -> None:
    """
    Set random seed for multiple libraries.

    Args:
        seed (int): a number chosen to see the random initialization.
    """
    LOG.info("Set random seed: {seed}")
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

def plot_train_curve(modelFilepath, trainInfo):
    train_losses = [summary['loss'] for summary in trainInfo[TrainStage.TRAIN.name]]
    valid_losses = [summary['loss'] for summary in trainInfo[TrainStage.VALIDATE.name]]
    # Create a range of epochs for x-axis
    trainepochs = range(1, len(train_losses) + 1)
    validepochs = range(1, len(valid_losses) + 1)

    # Create subplots
    fig = plt.figure(figsize=(12, 4))

    # Plot training loss on the left subplot
    plt.subplot(1, 2, 1)
    plt.plot(trainepochs, train_losses, label='Training Loss', color='b')
    plt.title('Training Loss Curve')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)

    # Plot validation loss on the right subplot
    plt.subplot(1, 2, 2)
    plt.plot(validepochs, valid_losses, label='Validation Loss', color='r')
    plt.title('Validation Loss Curve')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)

    plt.savefig(modelFilepath / 'loss_curve.png')  # Save the plot to a file
    plt.close(fig)


