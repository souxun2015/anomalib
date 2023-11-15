# Simplenet

This is the implementation of the [Simplenet](https://arxiv.org/abs/2303.15140) paper. It is based on https://github.com/DonaldRR/SimpleNet/tree/main and https://github.com/amazon-science/patchcore-inspection

To be noted, the noise adding method is different with the original method. In this implementation, we apply a adaptive noise adding method.
Model Type: Segmentation

## Description

SimpleNet comprises four parts: a pre-trained Feature Extractor, a Feature Adapter for transferring features to the target domain, an Anomaly Feature Generator that creates synthetic anomaly features, and an Anomaly Discriminator for differentiating between normal and anomaly features. Notably, the Anomaly Feature Generator is not used during inference.

## Usage

`python tools/train.py --model simplenet`

## Benchmark

All results gathered with seed `42`.

## [MVTec AD Dataset](https://www.mvtec.com/company/research/datasets/mvtec-ad)

### Image-Level AUC

|           |  Avg   | Carpet |  Grid  | Leather | Tile |  Wood  | Bottle | Cable  | Capsule | Hazelnut | Metal Nut |  Pill  | Screw  | Toothbrush | Transistor | Zipper |
| --------- | :----: | :----: | :----: | :-----: | :--: | :----: | :----: | :----: | :-----: | :------: | :-------: | :----: | :----: | :--------: | :--------: | :----: |
| Simplenet | 0.9836 | 0.9831 | 0.9974 |   1.0   | 1.0  | 0.9894 |  1.0   | 0.9835 | 0.9896  |  0.9967  |    1.0    | 0.9806 | 0.9647 |   0.8861   |   0.9979   | 0.9863 |

### Image F1 Score

|           |  Avg   | Carpet |  Grid  | Leather | Tile |  Wood  | Bottle | Cable  | Capsule | Hazelnut | Metal Nut |  Pill  | Screw  | Toothbrush | Transistor | Zipper |
| --------- | :----: | :----: | :----: | :-----: | :--: | :----: | :----: | :----: | :-----: | :------: | :-------: | :----: | :----: | :--------: | :--------: | :----: |
| Simplenet | 0.9788 | 0.9775 | 0.9911 |   1.0   | 1.0  | 0.9747 |  1.0   | 0.9444 | 0.9813  |  0.9855  |    1.0    | 0.9710 | 0.9460 |   0.9523   |   0.9750   | 0.9831 |

### Pixel-Level AUC

|           |  Avg   | Carpet |  Grid  | Leather |  Tile  |  Wood  | Bottle | Cable  | Capsule | Hazelnut | Metal Nut |  Pill  | Screw  | Toothbrush | Transistor | Zipper |
| --------- | :----: | :----: | :----: | :-----: | :----: | :----: | :----: | :----: | :-----: | :------: | :-------: | :----: | :----: | :--------: | :--------: | :----: |
| Simplenet | 0.9745 | 0.9834 | 0.9868 | 0.9868  | 0.9411 | 0.9212 | 0.9819 | 0.9700 | 0.9908  |  0.9824  |   0.99    | 0.9859 | 0.9917 |   0.9875   |   0.9331   | 0.9849 |
