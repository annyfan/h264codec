import argparse
import csv
import logging
import os
import sys
import traceback
from typing import TypeVar

import torch

from h264.src.dataload.dataloader_csv import create_dataset, get_test_dataloader
from h264.src.models.h264_model_bytes_ae_dalle import ModelTrainer, create_model
from h264.src.stages.utils import (
    set_random_seed,
    setup_config,
)
from tqdm import tqdm
from torchvision.utils import save_image

# Data type alias.
TDataLoader = TypeVar("TDataLoader", bound="torch.utils.data.dataloader.DataLoader")

# Set the logging for this application.
LOG = logging.getLogger(os.path.basename(__file__))


def eval_model(
    model_trainer: ModelTrainer,
    testLoader: TDataLoader,
    config: dict
) -> None:

    model_filepath = config["train"]["model_path"]

    # modelData.model.set_mode("train")
    model_trainer.eval()
    loadedData = None
    testIter = iter(testLoader)
    for loadedData in tqdm(testIter):
        with torch.no_grad():
            sample_res = model_trainer.sample(loadedData,retloss = True)
            genimages = sample_res["recon_x"]# returns
            originalimages = loadedData["image"].to(config["model"]["device"], non_blocking=True)
            for index, aimage in enumerate(genimages):
                outputimage = torch.cat((originalimages[index], aimage), 1)
                save_image(outputimage, model_filepath / f'./tsample-{loadedData["id"][index]}.png')


def setup_and_train(config: dict, baseDir: str = ".") -> None:
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
    dataset = create_dataset(config, evaluate = True)

    # Get the dataset loaders.
    testLoader = get_test_dataloader(config, dataset)

    LOG.info(
        "Test set > samples: %d, batches: %d,batch_size: %d",
        len(testLoader.dataset),
        len(testLoader),
        testLoader.batch_size,
    )
    model_trainer = create_model(config)
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
        eval_model(model_trainer, testLoader,config)
    except Exception as err:
        LOG.info(f"error in model training process {err}")
        traceback.print_exc()
    LOG.info("Completed the model training process.")


if __name__ == "__main__":
    argsParser = argparse.ArgumentParser()
    argsParser.add_argument("--config", dest="config", required=True)
    argsParser.add_argument("--basedir", dest="basedir", required=True)
    args = argsParser.parse_args()

    config = setup_config(args.config, args.basedir)
    setup_and_train(config=config, baseDir=args.basedir)
