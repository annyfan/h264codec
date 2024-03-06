import argparse
import logging
import os
import pprint
import sys
import time
import traceback
from typing import TypeVar

import numpy as np
import torch
import wandb
from einops import rearrange
from h264.src.dataload.dataloader_csv import create_dataset, set_dataloader
from h264.src.models.h264_model_bytes_ae_dalle import ModelTrainer,compute_loss, create_model
from h264.src.stages.utils import (
    create_batch_stats,
    create_epoch_summary,
    jit_save_model,
    perf_eval_epoch,
    plot_train_curve,
    set_random_seed,
    setup_config,
    TrainStage
)
import matplotlib.pyplot as plt
from torchmetrics.image.fid import FrechetInceptionDistance
import torchvision.transforms.functional as F
from torchvision.utils import save_image
# Data type alias.
TDataLoader = TypeVar("TDataLoader", bound="torch.utils.data.dataloader.DataLoader")

# Set the logging for this application.
LOG = logging.getLogger(os.path.basename(__file__))


def collect_inputs_and_compute_loss(
    data_trainer: ModelTrainer,  loadedData: dict, flagEval: bool, config: dict,  epoch_idx:int, iterationIdx: int
) -> float:
    """_summary_

    Args:
        data_trainer (ModelSummary): _description_
        loaded_data (dict): _description_
        flagEval (bool): _description_
        config (dict): _description_

    Returns:
        float: _description_
    """
    loss, loss_recons= compute_loss(data_trainer, loadedData, config)

    if not flagEval:
       data_trainer.update(epoch_idx , iterationIdx)

    return loss, loss_recons


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
        "last_epoch": config["train"].get('last_epoch', 0),
        "best_epoch": 0,
        "best_loss": float("inf"),
        "num_train_batches": len(trainLoader),
        "num_val_batches": len(valLoader),
        "num_epochs": config["train"]["num_epochs"],
        "batch_size": config["train"]["batch_size"],
    }

    LOG.debug(pprint.pformat(data, indent=4))
    return data

def log_images_table(table, generate_image, original_image, id):
    table.add_data(wandb.Image(generate_image), wandb.Image(original_image), id)
    
def sample(dataLoader:TDataLoader, model_trainer:ModelTrainer, epochIdx:int, model_filepath:str, imagename: str, metric, epochinfo):
        sampleLoaderIter = iter(dataLoader)
        sample_data = next(sampleLoaderIter)
        sample_data['image'] =  sample_data['image'][:16]
        sample_data['h264'] = sample_data['h264'][:16]
        sample_data["id"] = sample_data['id'][:16]

        sample_res = model_trainer.sample(sample_data)
        randomgenimages  = sample_res["recon_x"]# returns
        originalimage = sample_data['image']
        #for i in range(0,len(sample_data['h264'])):
                     
            #save_image(genimages[i], model_filepath / f'./tsample-{epochIdx}-{sample_data["id"][i]}.png')
            #save_image(sample_data['image'][i], model_filepath / f'./original-{epochIdx}-{sample_data["id"][i]}.png')

        if len(sample_data) > 0 and len(randomgenimages) > 0:
            epochinfo[imagename]=[wandb.Image(img.cpu().numpy().transpose(1, 2, 0)) for i, img in enumerate(originalimage)]  +  [wandb.Image(img.cpu().numpy().transpose(1, 2, 0)) for img in randomgenimages]   
        
        if metric is not None and len(randomgenimages) > 1:
            metric.update(randomgenimages.cpu().data.to(torch.uint8), real=True)
            metric.update(originalimage.cpu().data.to(torch.uint8), real=False)
            epochinfo['FID'] = metric.compute()
            metric.reset()

def sampletest(sample_data, model_trainer:ModelTrainer, epochIdx:int, model_filepath:str, imagename: str, metric, epochinfo):

        sample_res = model_trainer.sample(sample_data) 
        randomgenimages  = sample_res["recon_x"]# returns
        originalimage = sample_data['image']

        if len(sample_data) > 0 and len(randomgenimages) > 0:
            epochinfo[imagename]=[wandb.Image(img.cpu().numpy().transpose(1, 2, 0), caption=sample_data["id"][i]) for i, img in enumerate(originalimage)]  +  [wandb.Image(img.cpu().numpy().transpose(1, 2, 0)) for img in randomgenimages]   
        
        if metric is not None and len(randomgenimages) > 1:
            metric.update(randomgenimages.cpu().data.to(torch.uint8), real=True)
            metric.update(originalimage.cpu().data.to(torch.uint8), real=False)
            epochinfo['FID'] = metric.compute()
            metric.reset()

def samples(testLoader: TDataLoader):
    sampleLoaderIter = iter(testLoader)
    testsamples = next(sampleLoaderIter)
    testsamples['image'] =  testsamples['image'][:20]
    testsamples['h264'] = testsamples['h264'][:20]
    testsamples["id"] = testsamples['id'][:20]
    testsamples["id"] = ["train" if id in config["dataset"]["traintest"] else "valid" for id in testsamples["id"]]

    return testsamples

def train_model(
    model_trainer: ModelTrainer,
    trainLoader: TDataLoader,
    valLoader: TDataLoader,
    testLoader: TDataLoader,
    config: dict,
) -> None:
    currentLearningRate = 0.0
    trainInfo = get_traininfo(config, trainLoader, valLoader)
    model_filepath = config["train"]["model_path"]
    # Main training-validation loop.
    wandbtable = wandb.Table(columns=['gen_image', 'ori_image', 'id'])
    metric = FrechetInceptionDistance(feature=2048)
    trainiterationIdx = (trainInfo["last_epoch"]) * len(trainLoader)
    testsamples = samples(testLoader)
    for epochIdx in range(trainInfo["last_epoch"]+1, trainInfo["num_epochs"]+1):
        ### 1) Start with the training iterations.
        
        epochinfo = {}
        timeStart = time.time()
        trainInfo["timestamp"].append(timeStart)
        runningStats = np.zeros(trainInfo["num_train_batches"])
        runningStats2 = np.zeros(trainInfo["num_train_batches"])


        # modelData.model.set_mode("train")
        model_trainer.train()
        loadedData = None
        trainLoaderIter = iter(trainLoader)
        for batchIdx, loadedData in enumerate(trainLoaderIter):
            # Compute one model iteration.
            lossValue, loss_recons = collect_inputs_and_compute_loss(
                model_trainer, loadedData, False, config, epochIdx, trainiterationIdx + batchIdx
            )
            runningStats[batchIdx] = lossValue
            runningStats2[batchIdx] = loss_recons
            # Progress message.
            if batchIdx > 0 and batchIdx % config["train"]["show_epochs"] == 0:
                LOG.info(
                    create_batch_stats(runningStats, trainInfo, epochIdx, batchIdx,runningStats2)
                )
        
        trainiterationIdx += len(trainLoader)
        
        sample(trainLoader, model_trainer, epochIdx, model_filepath, 'train samples', None,epochinfo)

        # Create a summary for the training epoch.
        trainInfo[TrainStage.TRAIN.name].append(
            create_epoch_summary(
                runningStats,
                epochIdx,
                time.time() - timeStart,
                model_trainer.get_lr(),
                runningStats2
            )
        )
        epochinfo['trainloss'] = trainInfo[TrainStage.TRAIN.name][-1]['loss']
        epochinfo['trainloss_recons'] = trainInfo[TrainStage.TRAIN.name][-1]['loss2']

        ### 2) At the end of each epoch, obtain the validation loss.
        timeStart = time.time()
        runningStats = np.zeros(trainInfo["num_val_batches"])
        runningStats2 = np.zeros(trainInfo["num_val_batches"])

        # modelData.model.set_mode("eval")
        model_trainer.eval()
        valLoaderIter = iter(valLoader)
        valiterationIdx = (trainInfo["last_epoch"] + 1) * len(valLoader)
        for batchIdx, loadedData in enumerate(valLoaderIter):
            with torch.no_grad():
                lossValue, loss_recons = collect_inputs_and_compute_loss(
                    model_trainer, loadedData, True, config, epochIdx, valiterationIdx + batchIdx
                )
                runningStats[batchIdx] = lossValue
                runningStats2[batchIdx] = loss_recons

        sample(valLoader, model_trainer, epochIdx, model_filepath, 'validation samples', None, epochinfo)

        trainInfo[TrainStage.VALIDATE.name].append(
            create_epoch_summary(
                runningStats,
                epochIdx,
                time.time() - timeStart,
                currentLearningRate,
                loss_recons
            )
        )
        epochinfo['validloss'] = trainInfo[TrainStage.VALIDATE.name][-1]['loss']
        epochinfo['validloss_recons'] = trainInfo[TrainStage.VALIDATE.name][-1]['loss2']

        if not (epochIdx % config["train"]["sample_every"]) or epochIdx == trainInfo[
            "num_epochs"] :  # and trainer.is_main:  # is_main makes sure this can run in distributed
            sampletest(testsamples, model_trainer, epochIdx, model_filepath, 'test samples', None, epochinfo)

            
        ### 3) Adjust the learning rates based on losses.
        bestModelFlag = False
        if trainInfo["num_val_batches"] > 0:
            #model_trainer.step(trainInfo[TrainStage.VALIDATE.name][-1]["loss"])
            if (
                trainInfo[TrainStage.VALIDATE.name][-1]["loss"]
                <= trainInfo["best_loss"]
            ):
                trainInfo["best_loss"] = trainInfo[TrainStage.VALIDATE.name][-1]["loss"]
                trainInfo["best_epoch"] = epochIdx
                bestModelFlag = True
        else:
            #modelData.scheduler.step(trainInfo[TrainStage.TRAIN.name][-1]["loss"])
            if trainInfo[TrainStage.TRAIN.name][-1]["loss"] <= trainInfo["best_loss"]:
                trainInfo["best_loss"] = trainInfo[TrainStage.TRAIN.name][-1]["loss"]
                trainInfo["best_epoch"] = epochIdx
                bestModelFlag = True

        ### 4) Update the current learning rate.

        if  config['train']['scheduler']['name'] in  ["plateau"] :
            reduce_scheduler = model_trainer.scheduler
            reduce_scheduler.update_lr(
                optimizer=model_trainer.optimizer, 
                epoch=epochIdx, 
                curr_iter=trainiterationIdx, 
                current_loss = trainInfo[TrainStage.TRAIN.name][-1]["loss"]
            )
            

        currentLearningRate = model_trainer.get_lr()
        epochinfo['currentLearningRate'] = currentLearningRate

        wandb.log(epochinfo,  step=epochIdx)    
        ### 5) Print the epoch results, save the best model and render the training curves
        LOG.info(perf_eval_epoch(trainInfo, config["train"]["model_path"], TrainStage.TRAIN, epochIdx))
        LOG.info(perf_eval_epoch(trainInfo, config["train"]["model_path"], TrainStage.VALIDATE, epochIdx))

        if bestModelFlag and epochIdx >=  5 :
            jitFilename = model_filepath / "best_model.pt"
            LOG.info(
                "Saving the current best model from epoch %d to %s.",
                epochIdx,
                jitFilename,
            )
            if loadedData:
                model_trainer.save(jitFilename)

        ### 6) Plot train curve and overwrite the latest model.
        if epochIdx > 1:
            LOG.info(
                "Epoch %d - Best epoch: %d, loss: %f.",
                epochIdx,
                trainInfo["best_epoch"],
                trainInfo["best_loss"],
            )
            plot_train_curve(model_filepath, trainInfo)
            if loadedData and not (epochIdx % config["train"]["checkpoint_every"]):
                checkpoint_path = model_filepath / f'checkpoint.{epochIdx}.pt'
                model_trainer.save_to_checkpoint_folder(checkpoint_path=checkpoint_path)

    wandb.log({"best_epoch":  trainInfo["best_epoch"], "best_loss": trainInfo["best_loss"]}, commit=False)
    wandb.log({"predictions_table":wandbtable}, commit=False)

    if loadedData:
        filepath = os.path.join(model_filepath, f'checkpoint.{trainInfo["num_epochs"]}.pt')
        model_trainer.save(filepath)


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
    model_trainer = create_model(config)

    
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
        train_model(model_trainer, trainLoader, valLoader, testLoader, config)
    except Exception as err:
        LOG.info(f"error in model training process {err}")
        traceback.print_exc()
    LOG.info("Completed the model training process.")


if __name__ == "__main__":
    argsParser = argparse.ArgumentParser()
    argsParser.add_argument("--config", dest="config", required=True)
    argsParser.add_argument("--basedir", dest="basedir", required=True)
    argsParser.add_argument("--resume", dest="resume", required=False)
    args = argsParser.parse_args()

    config = setup_config(args.config, args.basedir)
    if args.resume is not None:
        with wandb.init(project="pytorch-byteformer-imagen", config=config, id=args.resume, resume="allow") as run:
            setup_and_train(config=config, baseDir=args.basedir)
    else:
        with wandb.init(project="pytorch-byteformer-imagen", config=config) as run:
            setup_and_train(config=config, baseDir=args.basedir)


