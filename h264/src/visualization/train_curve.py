# ----------------------------------------------------------------------------------------------
# Copyright (C) B - All Rights Reserved
# Unauthorized copying, use, or modification to this file via any medium is strictly prohibited.
# This file is private and confidential.
# Contact: dev@b
# ----------------------------------------------------------------------------------------------

import logging
import pathlib
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from h264.src.stages.utils import TrainStage

logger = logging.getLogger("TrainCurve")

matplotlib.use("agg")


def plot_single_loss(config: dict, trainInfo: dict) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(12, 14))
    # plot batch curve
    for phase in [TrainStage.TRAIN, TrainStage.VALIDATE]:
        epochs = [t["epoch"] for t in trainInfo[phase.name]]
        color = "darkorange" if phase == TrainStage.VALIDATE else "royalblue"

        # Loss
        loss = [t["loss"] for t in trainInfo[phase.name]]
        axes[0].plot(
            epochs, loss, "-", label=phase.name, color=color, linewidth=1.5, alpha=0.8
        )
        if epochs[-1] < 100:
            axes[0].plot(epochs, loss, ".", label="", color=color, alpha=0.8)
        idxLoss = np.argmin(loss)
        axes[0].annotate(
            f"{loss[idxLoss]:.5f}",
            (epochs[idxLoss], loss[idxLoss]),
            xytext=(-15, 30),
            textcoords="offset pixels",
            arrowprops={"arrowstyle": "->"},
        )

        # Learning rate
        if phase == TrainStage.TRAIN:
            learnRate = [t["lr"] for t in trainInfo[phase.name]]
            axes[1].plot(
                epochs,
                learnRate,
                "-",
                label="model",
                color="yellowgreen",
                linewidth=1.5,
                alpha=0.8,
            )
            if epochs[-1] < 100:
                axes[1].plot(
                    epochs, learnRate, ".", label="", color="yellowgreen", alpha=0.8
                )

    # Add the title, legend and axis labels.
    axes[0].legend(loc="upper right")
    axes[0].set_xlabel("epoch")
    axes[0].set_ylabel("Loss")
    axes[0].grid(which="major", color="#DDDDDD", linewidth=0.8)
    axes[0].grid(which="minor", color="#EEEEEE", linestyle=":", linewidth=0.5)
    axes[0].minorticks_on()

    axes[1].legend(loc="upper left")
    axes[1].set_xlabel("epoch")
    axes[1].set_ylabel("Learning rate")
    axes[1].grid(which="major", color="#DDDDDD", linewidth=0.8)
    axes[1].grid(which="minor", color="#EEEEEE", linestyle=":", linewidth=0.5)
    axes[1].minorticks_on()

    # Save as a PNG figure, but first create a descriptive title.
    trainTime = np.mean([t["time"] for t in trainInfo[TrainStage.TRAIN.name]])
    valTime = np.mean([t["time"] for t in trainInfo[TrainStage.VALIDATE.name]])
    path = pathlib.PurePath(config["train"]["model_path"])
    titleMsg = (
        r"$\bf{BytesNet\ -\ B}$"
        f"\n{path.name}\n"
        f"Time[T/V](s):{trainTime:.1f}/{valTime:.1f}"
    )
    fig.suptitle(titleMsg)
    plt.tight_layout()
    filename = config["train"]["model_path"] / "training_curve.png"
    plt.savefig(filename, dpi=100)
    plt.clf()
    plt.close()


def plot_multi_loss(config: dict, trainInfo: dict) -> None:
    fig, axes = plt.subplots(3, 1, figsize=(12, 14))
    # plot batch curve
    for phase in [TrainStage.TRAIN, TrainStage.VALIDATE]:
        epochs = [t["epoch"] for t in trainInfo[phase.name]]
        color = "darkorange" if phase == TrainStage.VALIDATE else "royalblue"

        # Contrastive loss
        loss = [t["loss_wav2vec"] for t in trainInfo[phase.name]]
        axes[0].plot(
            epochs, loss, "-", label=phase.name, color=color, linewidth=1.5, alpha=0.8
        )
        if epochs[-1] < 100:
            axes[0].plot(epochs, loss, ".", label="", color=color, alpha=0.8)
        idxLoss = np.argmin(loss)
        axes[0].annotate(
            f"{loss[idxLoss]:.5f}",
            (epochs[idxLoss], loss[idxLoss]),
            xytext=(-15, 30),
            textcoords="offset pixels",
            arrowprops={"arrowstyle": "->"},
        )

        # Hype loss
        loss = [t["loss_hype"] for t in trainInfo[phase.name]]
        axes[1].plot(
            epochs, loss, "-", label=phase.name, color=color, linewidth=1.5, alpha=0.8
        )
        if epochs[-1] < 100:
            axes[1].plot(epochs, loss, ".", label="", color=color, alpha=0.8)
        idxLoss = np.argmin(loss)
        axes[1].annotate(
            f"{loss[idxLoss]:.5f}",
            (epochs[idxLoss], loss[idxLoss]),
            xytext=(-15, 30),
            textcoords="offset pixels",
            arrowprops={"arrowstyle": "->"},
        )

        # Learning rate
        if phase == TrainStage.TRAIN:
            learnRate = [t["lr_wav2vec"] for t in trainInfo[phase.name]]
            axes[2].plot(
                epochs,
                learnRate,
                "-",
                label="wav2vec",
                color="yellowgreen",
                linewidth=1.5,
                alpha=0.8,
            )
            if epochs[-1] < 100:
                axes[2].plot(
                    epochs, learnRate, ".", label="", color="yellowgreen", alpha=0.8
                )

            learnRate = [t["lr_hype"] for t in trainInfo[phase.name]]
            axes[2].plot(
                epochs,
                learnRate,
                "-",
                label="hype",
                color="royalblue",
                linewidth=1.5,
                alpha=0.8,
            )
            if epochs[-1] < 100:
                axes[2].plot(
                    epochs, learnRate, ".", label="", color="royalblue", alpha=0.8
                )

    # Add the title, legend and axis labels.
    axes[0].legend(loc="upper right")
    axes[0].set_xlabel("epoch")
    axes[0].set_ylabel("Contrast Loss")
    axes[0].grid(which="major", color="#DDDDDD", linewidth=0.8)
    axes[0].grid(which="minor", color="#EEEEEE", linestyle=":", linewidth=0.5)
    axes[0].minorticks_on()

    axes[1].legend(loc="upper right")
    axes[1].set_xlabel("epoch")
    axes[1].set_ylabel("Hype Loss")
    axes[1].grid(which="major", color="#DDDDDD", linewidth=0.8)
    axes[1].grid(which="minor", color="#EEEEEE", linestyle=":", linewidth=0.5)
    axes[1].minorticks_on()

    axes[2].legend(loc="upper left")
    axes[2].set_xlabel("epoch")
    axes[2].set_ylabel("Learning rate")
    axes[2].grid(which="major", color="#DDDDDD", linewidth=0.8)
    axes[2].grid(which="minor", color="#EEEEEE", linestyle=":", linewidth=0.5)
    axes[2].minorticks_on()

    # Save as a PNG figure, but first create a descriptive title.
    trainTime = np.mean([t["time"] for t in trainInfo[TrainStage.TRAIN.name]])
    valTime = np.mean([t["time"] for t in trainInfo[TrainStage.VALIDATE.name]])
    path = pathlib.PurePath(config["train"]["model_path"])
    titleMsg = (
        r"$\bf{HypeNet\ -\ B}$"
        f"\n{path.name}\n"
        f"Time[T/V](s):{trainTime:.1f}/{valTime:.1f}"
    )
    fig.suptitle(titleMsg)
    plt.tight_layout()
    filename = config["train"]["model_path"] / "training_curve.png"
    plt.savefig(filename, dpi=100)
    plt.clf()
    plt.close()


def plot_train_curve(config: dict, trainInfo: dict) -> None:
    """
    Plot training curves.

    Args:
        config:
            The global parameters configuration.

        trainInfo:
            The dictionary of all training and validation statistics for all iterations.
    """
    if len(trainInfo[TrainStage.TRAIN.name]) == 0:
        logger.warning("Model info %s has no data", config["train"]["model_path"])
        return

    # Initialize plot elements.
    if "loss" in trainInfo[TrainStage.TRAIN.name][-1]:
        plot_single_loss(config, trainInfo)
    elif "loss_wav2vec" in trainInfo[TrainStage.TRAIN.name][-1]:
        plot_multi_loss(config, trainInfo)
    else:
        raise ValueError("Invalid dict keys")
