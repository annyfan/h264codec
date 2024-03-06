# rnd-codec-dev
RnD Codec


# Deep architecture for H.264 I-Frame decoding

This repository is the implementation of Deep architecture for H.264 decoding. 


## Requirements

To install requirements:

```setup
pip install -r requirements.txt
```

To download dataset:
TODO:


Data preprocess: 

- open h264/src/utils/cleanup_samples.py, make sure the dataset folders are correct
- run: 
```cleanup
python3 h264/src/utils/cleanup_samples.py
```


## Training

To train the model(s) in the paper, run this command:

```train
python3 h264/src/stages/train_h264_ae_dalle.py --config params_bytes_ae_mssiml1_vgg.yaml --basedir h264
```
!You need to enter the wandb password.

## Sample data

To generate samples on Test Set, run:

```eval
python h264/src/stages/eval_h264_ae_dalle.py --config h264/params_bytes_ae_eval.yaml --basedir h264
```
!You can configure test samples in config.yaml/dataset/traintest, config.yaml/dataset/validtest or dataset.csv.

## Pre-trained Models

You can download pretrained models here:

- [model 1](https://drive.google.com/drive/folders/10QZ046i6mrdL5v0RKaTCFxpUray_De5z?usp=drive_link) trained 800 epochs on 1.8k(duplicated samples) dataset h264_v20231127. 
- [model 2](https://drive.google.com/drive/folders/10NiUee4a1F4FAIf3xtsSx2EbfMskVGOd?usp=drive_link) trained 500 epochs on 1.3k dataset h264_v20231127. 
- [model 3](https://drive.google.com/drive/folders/10raW7MKMVcWnJj6y2ekMaVNkNFFl0M3J?usp=drive_link) trained 250 epochs on 1M dataset h264_v20231127, h264_v20240206_1, h264_v20240206_2, h264_v20240206_3, h264_v20240206_4, h264_v20240206_5, h264_v20240206_6, h264_v20240206_7. 

## Results

Our model achieves the following performance on :


| Model name         |    epochs       | Training Loss  | Validation Loss|           Dataset           |
| -----------------  |---------------- | -------------- | -------------- |-----------------------------|
| model 1            |     800         |      7.93      |     30.632     |   1.8k(duplicated samples)  |
| model 2            |     500         |      12.863    |     30.448     |            1.3k             |
| model 3            |     250         |      20.03     |     27.348     |             1M              |


#Related Work
https://github.com/apple/ml-cvnets
https://github.com/AntixK/PyTorch-VAE/blob/master/models/mssim_vae.py#L182
https://github.com/crowsonkb/vgg_loss
https://github.com/psyrocloud/MS-SSIM_L1_LOSS/tree/main
https://github.com/openai/DALL-E
