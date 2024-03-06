import numpy as np
from dataclasses import dataclass
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from typing import List, Tuple, Union

from h264.src.models.modules.byteformer.byteformer import ByteFormer
from h264.src.scheduler.cosine import CosineScheduler
from h264.src.criterion.jsd_cross_entropy import JsdCrossEntropy


@dataclass
class ModelSummary:
    """Returned elements for the model building."""

    model: torch.nn.Module = None
    optimizer: Union[torch.optim.Adam, torch.optim.Adagrad] = None
    scheduler: torch.optim.lr_scheduler.ReduceLROnPlateau = None
    scaler: torch.cuda.amp.GradScaler = None
    criterion: torch.nn.Module = None


def compute_loss(
    modelData: ModelSummary, data: Tuple[torch.Tensor, torch.Tensor], config: dict
) -> torch.Tensor:
    """Calls the model and computes the loss."""
    histo = data["histo"].to(config["model"]["device"], non_blocking=True)
    h264 = data["h264"].to(config["model"]["device"], non_blocking=True)
    results = modelData.model(h264)

    # Compute presence loss (order is important).
    loss = modelData.criterion(results, histo)

    return loss


# Create H264 model
def create_model(config: dict) -> ModelSummary:
    """Set the model and training components."""
    model = HistoNet(config)

    # Scheduler
    if config["train"]["scheduler"] == "plateau":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            factor=config["train"]["decay"],
            patience=config["train"]["patience"],
            min_lr=1e-7,
            verbose=True,
            threshold_mode="abs",
        )
        lr = config["train"]["learning_rate"]
    elif config["train"]["scheduler"] == "cosine":
        scheduler = CosineScheduler(
            warmup_iterations=config["train"]["scheduler_cosine"]["warmup_iterations"],
            warmup_init_lr=config["train"]["scheduler_cosine"]["warmup_init_lr"],
            max_iterations=config["train"]["scheduler_cosine"]["max_iterations"],
            min_lr=config["train"]["scheduler_cosine"]["min_lr"],
            max_lr=config["train"]["scheduler_cosine"]["max_lr"],
        )
        lr = scheduler.get_lr(0, 0)
    else:
        raise ValueError(f'Invalid training scheduler {config["train"]["scheduler"]}.')

    # Optimizer.
    if config["train"]["optimizer"] == "Adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, amsgrad=True)
    else:
        optimizer = torch.optim.Adagrad(model.parameters(), lr=lr)

    # create scaler for AMP - AUTOMATIC MIXED PRECISION PACKAGE
    scaler = torch.cuda.amp.GradScaler()

    # Criterion
    if config["train"]["criterion"] == "l1":
        criterion = torch.nn.L1Loss(reduction="mean")
    elif config["train"]["criterion"] == "mse":
        criterion = torch.nn.MSELoss(reduction="mean")
    elif config["train"]["criterion"] == "jse":
        criterion = JsdCrossEntropy(num_splits=1, alpha=12, smoothing=0.0)
    else:
        raise ValueError(f'Invalid training criterion {config["train"]["criterion"]}.')

    # Return the model, the optimizer, the scheduler and scaler.
    return ModelSummary(
        model=model.to(config["model"]["device"]),
        optimizer=optimizer,
        scheduler=scheduler,
        scaler=scaler,
        criterion=criterion,
    )


class HistoNet(nn.Module):
    """
    Model class ...
    """

    def __init__(self, config: dict) -> None:
        super(HistoNet, self).__init__()

        self.outputImageSize = config["dataset"]["img_size"]
        self.device = config["model"]["device"]

        # Define the byteFormer network.
        self.byteFormer = ByteFormer(config)

        # Common linear model at the tail for the backbone network.
        self.regressor = nn.Sequential(
            nn.Linear(
                config["model"]["byteformer"]["embed_dim"],
                config["dataset"]["hist_bins"],
            ),
            nn.BatchNorm1d(config["dataset"]["hist_bins"], momentum=0.1),
            nn.ReLU(),
        )

    def forward(self, inputData: torch.Tensor) -> torch.Tensor:
        """FOO"""
        emb = self.byteFormer(inputData)
        res = self.regressor(emb["c"])

        return res
