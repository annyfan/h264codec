import argparse
import csv
import logging
import os
import sys
import traceback
from typing import TypeVar

import torch

from h264.src.dataload.dataloader_csv import create_dataset, set_dataloader
from h264.src.models.h264_model_bytes_ae_dalle import ModelTrainer,compute_loss, create_model
from h264.src.stages.utils import (
    set_random_seed,
    setup_config,
)
from pathlib import Path
from tqdm import tqdm
from torchvision.utils import save_image
from h264.src.dataload.utils import SubsetSequentialSampler
import pandas as pd

# Data type alias.
TDataLoader = TypeVar("TDataLoader", bound="torch.utils.data.dataloader.DataLoader")

# Set the logging for this application.
LOG = logging.getLogger(os.path.basename(__file__))


def infer_model(
    model_trainer: ModelTrainer,
    trainLoader: TDataLoader,
    valLoader: TDataLoader,
    testLoader: TDataLoader,
    config: dict,
    gencsv: bool
) -> None:

    model_filepath = config["train"]["model_path"]

    # modelData.model.set_mode("train")
    model_trainer.eval()
    loadedData = None
    trainLoaderIter = iter(trainLoader)
    if gencsv:
        with open(Path(model_filepath) /  "dataset_info.csv",  mode='w') as f:
            writer = csv.writer(f)
            writer.writerow(['id','flag', 'mse','size'])
            
            for loadedData in tqdm(trainLoaderIter):
                # Compute one model iteration.
                with torch.no_grad():
                    sample_res = model_trainer.sample(loadedData, retloss = True)
                    losses = sample_res["losses"]# returns
                    for index, loss in enumerate(losses):
                        writer.writerow([loadedData["id"][index],'TRAIN', loss.item(), loadedData["size"][index].item()])
            
            valLoaderIter = iter(valLoader)
            for loadedData in tqdm(valLoaderIter):
                with torch.no_grad():
                    sample_res = model_trainer.sample(loadedData,retloss = True)
                    losses = sample_res["losses"]# returns
                    for index, loss in enumerate(losses):
                        writer.writerow([loadedData["id"][index],'VALIDATE', loss.item(), loadedData["size"][index].item()])

    testIter = iter(testLoader)
    for loadedData in tqdm(testIter):
        with torch.no_grad():
            sample_res = model_trainer.sample(loadedData,retloss = True)
            genimages = sample_res["recon_x"]# returns
            originalimages = loadedData["image"].to(config["model"]["device"], non_blocking=True)
            for index, aimage in enumerate(genimages):
                outputimage = torch.cat((originalimages[index], aimage), 1)
                save_image(outputimage, model_filepath / f'./tsample-{loadedData["id"][index]}.png')


def setup_and_train(config: dict, baseDir: str = ".", gencsv = True) -> None:
    """
    Setup before calling the training function.

    Args:
        configPath (str): the path to the yaml configuration file.
        baseDir (str): the reference directory for the configuration links.
    """ 
    set_random_seed(config["base"]["random_seed"])

    # Set the logger configuration. 
    logging.basicConfig(
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        level=logging.getLevelName(config["base"]["logging"]),
        stream=sys.stdout,
    )
    LOG.info('config: %s', config)
    # Create a dataset.
    dataset = create_dataset(config)

    # Get the dataset loaders.
    trainLoader, valLoader, testLoader = set_dataloader(config, dataset)
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
    LOG.info(
        "Test set > samples: %d, batches: %d,batch_size: %d",
        len(testLoader.dataset),
        len(testLoader),
        testLoader.batch_size,
    )

    # Create a model and support optimization elements.
    byteformer_total_params = sum(
        param.numel() for param in model_trainer.model.module.byteformer.parameters()
    )

    bottleneck_total_params = sum(
        param.numel() for param in model_trainer.model.module.reduction_net.parameters()
    )

    deconv_total_params = sum(
        param.numel() for param in model_trainer.model.module.decoder.parameters()
    )

    LOG.info(f"byteformer parameters {byteformer_total_params}")
    LOG.info(f"bottleneck parameters {bottleneck_total_params}")
    LOG.info(f"deconv parameters {deconv_total_params}")

    if 'weights' in config['model']:
        model_trainer.load(config['model']['weights'], strict=True)

    if 'checkpoint' in config['model']:
        model_trainer.load_from_checkpoint_folder(config['model']['checkpoint'], config["train"]['last_epoch'])

        # Run the training phases.
    LOG.info("Starting the model training process.")
    try:
        infer_model(model_trainer, trainLoader, valLoader, testLoader,config, gencsv)
    except Exception as err:
        LOG.info(f"error in model training process {err}")
        traceback.print_exc()
    LOG.info("Completed the model training process.")


if __name__ == "__main__":
    argsParser = argparse.ArgumentParser()
    argsParser.add_argument("--config", dest="config", required=True)
    argsParser.add_argument("--basedir", dest="basedir", required=True)
    argsParser.add_argument("--no-csv",  dest='csv', action='store_false')
    args = argsParser.parse_args()

    config = setup_config(args.config, args.basedir)
    setup_and_train(config=config, baseDir=args.basedir, gencsv = args.csv)
