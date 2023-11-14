# EfficientAd

This is the implementation of the [Simplenet](https://arxiv.org/abs/2303.15140) paper. It is based on https://github.com/DonaldRR/SimpleNet/tree/main and https://github.com/amazon-science/patchcore-inspection

Model Type: Segmentation

## Description

SimpleNet comprises four parts: a pre-trained Feature Extractor, a Feature Adapter for transferring features to the target domain, an Anomaly Feature Generator that creates synthetic anomaly features, and an Anomaly Discriminator for differentiating between normal and anomaly features. Notably, the Anomaly Feature Generator is not used during inference.

## Usage

`python tools/train.py --model simplenet`

## Benchmark

All results gathered with seed `42`.

## [MVTec AD Dataset](https://www.mvtec.com/company/research/datasets/mvtec-ad)

### Image-Level AUC

|           | Avg | Carpet | Grid | Leather | Tile | Wood | Bottle | Cable | Capsule | Hazelnut | Metal Nut | Pill | Screw | Toothbrush | Transistor | Zipper |
| --------- | :-: | :----: | :--: | :-----: | :--: | :--: | :----: | :---: | :-----: | :------: | :-------: | :--: | :---: | :--------: | :--------: | :----: |
| Simplenet | 0.  |   0.   |  0.  |   0.    |  0.  |  0.  |   0.   |  0.   |   0.    |    0.    |    0.     |  0.  |  0.   |     0.     |     0.     |   0.   |

### Image F1 Score

|           | Avg | Carpet | Grid | Leather | Tile | Wood | Bottle | Cable | Capsule | Hazelnut | Metal Nut | Pill | Screw | Toothbrush | Transistor | Zipper |
| --------- | :-: | :----: | :--: | :-----: | :--: | :--: | :----: | :---: | :-----: | :------: | :-------: | :--: | :---: | :--------: | :--------: | :----: |
| Simplenet | 0.  |   0.   |  0.  |   0.    |  0.  |  0.  |   0.   |  0.   |   0.    |    0.    |    0.     |  0.  |  0.   |     0.     |     0.     |   0.   |
