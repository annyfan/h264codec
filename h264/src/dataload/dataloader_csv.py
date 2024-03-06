import csv
import logging
import random
from functools import partial
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torchvision
from PIL import Image
from torch.utils import data
from torch.utils.data import DataLoader, SubsetRandomSampler

from h264.src.dataload.byteformer_collate_functions import byteformer_h264_collate_fn
from h264.src.dataload.utils import SubsetSequentialSampler
from h264.src.stages.utils import TrainStage
from h264.src.dataload.byteformer_collate_functions import byteformer_h264_collate_fn

# LOG with namespace identifier.
LOG = logging.getLogger(__name__)
LOG.addHandler(logging.NullHandler())


def seed_worker(workerId: int) -> None:
    """Method for random values initialization in the dataloader. The argument list is fixed.

    Args:
        workerId: worker id.
    """
    workerSeed = torch.initial_seed() % 2 ** 32
    np.random.seed(workerSeed)
    random.seed(workerSeed)

def create_dataset(config: dict, evaluate: bool = False) -> "H264Data":
    """
    Utility function for the creation of a data loader, including
    loundness data, data normalization and transformation.

    Args:
        config:
            The dictionary with the application configuration.

    Returns:
        The HypeData data loader.
    """
    image_size = (config["dataset"]["img_size"][0], config["dataset"]["img_size"][1])

    # Create the data normalization
    imgNormalize = torchvision.transforms.Normalize(
        mean=config["dataset"]["img_norm_mean"], std=config["dataset"]["img_norm_std"]
    )
    if evaluate:
        imgTransform = torchvision.transforms.Compose(
            [
                torchvision.transforms.Resize(image_size),
                #torchvision.transforms.CenterCrop(image_size),
                torchvision.transforms.ToTensor(),
                #imgNormalize,
            ]
        )
    else:
        imgTransform = torchvision.transforms.Compose(
            [
                torchvision.transforms.Resize(image_size),
                #torchvision.transforms.ColorJitter(
                #    brightness=config["dataset"]["img_jitter"],
                #    contrast=config["dataset"]["img_jitter"],
                #    saturation=config["dataset"]["img_jitter"],
                #    hue=config["dataset"]["img_hue"],
                #),
                torchvision.transforms.ToTensor(),
                #AddGaussianNoise(
                #    config["dataset"]["img_noise"][0], config["dataset"]["img_noise"][1]
                #),
                #imgNormalize,
            ]
        )

    # Data tranformation.
    dataset = H264Data(config, imgTransform)

    return dataset


def set_dataloader(
    config: dict,
    dataset: "H264Data",
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Create the DataLoaders for training, validation and testing.

    Args:
        config:
            The dictionary with the application configuration.
        dataset:
            The deeplake dataset for training.

    Returns:
        A tuple of dataloaders (for training, validation and testing).
    """
    # Determine the type of sampler based on the configuration name. Invalid strings will trigger
    # a logger error with full stack information.
    if config["train"].get("sampler", "Random") == "Random":
        selsampler = SubsetRandomSampler
    elif config["train"].get("sampler", "Sequential") == "Sequential":
        selsampler = SubsetSequentialSampler
    else:
        selsampler = SubsetRandomSampler
        LOG.error("Invalid sampler selected.", exc_info=True)

    trainLoader = DataLoader(
        dataset,
        batch_size=config["train"]["batch_size"],
        sampler=selsampler(dataset.get_indices(TrainStage.TRAIN)),
        num_workers=config["train"]["num_workers"],
        worker_init_fn=seed_worker,
        pin_memory=True,
        collate_fn=partial(byteformer_h264_collate_fn, config=config),
    )
    valLoader = DataLoader(
        dataset,
        batch_size=config["train"]["batch_size"],
        sampler=selsampler(dataset.get_indices(TrainStage.VALIDATE)),
        num_workers=config["train"]["num_workers"],
        worker_init_fn=seed_worker,
        pin_memory=True,
        collate_fn=partial(byteformer_h264_collate_fn, config=config),
    )
    testLoader = DataLoader(
        dataset,
        batch_size=config["train"]["batch_size_test"],
        sampler=selsampler(dataset.get_indices(TrainStage.TEST)),
        num_workers=1,
        worker_init_fn=seed_worker,
        pin_memory=True,
        collate_fn=partial(byteformer_h264_collate_fn, config=config),
    )

    return trainLoader, valLoader, testLoader

def get_test_dataloader(
        config: dict,
        dataset: "H264Data",
    ) -> DataLoader:
    testLoader = DataLoader(
        dataset,
        batch_size=config["train"]["batch_size_test"],
        sampler=SubsetSequentialSampler(dataset.get_indices(TrainStage.TEST)),
        num_workers=1,
        worker_init_fn=seed_worker,
        pin_memory=True,
        collate_fn=partial(byteformer_h264_collate_fn, config=config),
    )
    return testLoader

def make_dataset(datasetNames: List[str], config: dict) -> Dict[str, List[str]]:
    """
    Parse the dataset files and create the list of files, detection and poses.
    """
    data = {
        "id": [],
        "dataset": [],
        "trainIndices": [],
        "testIndices": [],
        "valIndices": [],
    }

    index = 0
    for dataset in datasetNames:
        traintest =  config["dataset"].get("traintest", [])
        validtest =  config["dataset"].get("validtest", [])
        datasetcsv = config["dataset"].get("datasetcsv", "dataset.csv")
        with open(Path(config["dataset"]["base_dir"]) / dataset / datasetcsv) as f:
            reader = csv.reader(f)
            next(reader, None)  # skip the headers
            for row in reader:
                data["dataset"].append(dataset)
                data["id"].append(row[0])

                trainFlag = row[1]
                if trainFlag == TrainStage.TRAIN.name:
                    data["trainIndices"].append(index)
                elif trainFlag == TrainStage.VALIDATE.name:
                    data["valIndices"].append(index)
                elif trainFlag == TrainStage.TEST.name:
                    data["testIndices"].append(index)
                else:
                    raise ValueError("Invalid training flag.")
                
                if traintest is not None and row[0] in traintest:
                    data["testIndices"].append(index)
                if validtest is not None and row[0] in validtest:
                    data["testIndices"].append(index)

                index += 1

    return data


def image_loader(imgPath: Path):
    """
    Image dataloader from a file path and an alpha mask.

    Parameters
    ----------
    imgPath : string
        The full file path to the image.

    Returns
    -------
    The RGB image array.
    """
    img = Image.open(imgPath).convert("RGB")

    return img


def h264_loader(h264Path: Path):
    """
    TBD
    """
    h264 = torchvision.io.read_file(str(h264Path))
    #h264 = torch.Tensor(torch.frombuffer(torchvision.io.read_file(str(h264Path)),  dtype=torch.int32))
    return h264


class H264Data(data.Dataset):
    """Dataset class for loading the loudness data, given an index."""

    def __init__(
            self,
            config: dict,
            imgTransform: torchvision.transforms.Compose,
    ):
        """
        HypeData constructor.

        Args:
            config (dict):
                The application configuration dictionary.
            imgNormalize:
                The functor used to transform/normalize the model features.
            imgTransform:
                The functor used to transform/normalize the model features.

        Raises:
            ValueError: a value exception is raised if the training flag is invalid.
        """
        self.cfg = config

        # Get the data from the configuration files.
        datasetNames = [
            f"{self.cfg['dataset']['base_name']}_{name}"
            for name in self.cfg["dataset"]["versions"]
        ]
        self.dataset = make_dataset(datasetNames, config)

        # Data transformation (e.g. normalization)
        self.imgTransform = imgTransform
        self.h264PaddedSeqLen = config["dataset"].get("h264_padded_seq_len", None)

        # Base path
        self.basePath = Path(config["dataset"]["base_dir"])
        self.baseName = Path(config["dataset"]["base_name"])

    def get_indices(self, phase: TrainStage = TrainStage.TRAIN) -> List[int]:
        """
        Utility for getting a list of all indices, based on the phase id. A shuffle option is
        available for the training/validation phase.

        Args:
            phase:
                The training phase enumerated value.

        Returns:
           The list of indices for either the train/validate or test phases.
        """
        if phase == TrainStage.TRAIN:
            return self.dataset["trainIndices"]
        if phase == TrainStage.VALIDATE:
            return self.dataset["valIndices"]
        if phase == TrainStage.TEST:
            return self.dataset["testIndices"]

        raise ValueError

    def __getitem__(self, index: int) -> dict:
        """Get an item at a given index in dataset.

        Args:
            index:
                The list index for the dataset.

        Returns:
            A dictionary of values for training and testing.
        """
        # Load images
        imageFilename = (
                self.basePath
                / f"{self.dataset['dataset'][index]}"
                / "image"
                / f"{self.dataset['id'][index]}.jpeg"
        )
        image = self.imgTransform(image_loader(imageFilename))
        hist = torch.histogram(
            torch.mean(image, dim=0),
            bins=self.cfg["dataset"]["hist_bins"],
            density=False,
        )

        # Load h264
        h264Filename = (
                self.basePath
                / f"{self.dataset['dataset'][index]}"
                / "h264"
                / f"{self.dataset['id'][index]}.h264"
        )
        h264 = h264_loader(h264Filename)

        return {
            "image": image, #self.imgTransform(image),
            "histo": torch.nn.functional.normalize(hist.hist, p=1, dim=0),
            "h264":  h264, #torch.nn.functional.pad( h264, (0, self.h264PaddedSeqLen - h264.shape[0]), value=0 ),
            "id": self.dataset["id"][index],
            "size": h264.shape[0]
        }

    def __len__(self) -> int:
        """Return size (num. entries) of the dataset.

        Returns:
           The total number of dataset elements.
        """

        return len(self.dataset["id"])
