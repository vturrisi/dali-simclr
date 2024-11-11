# MAGMA: Manifold Regularization for MAEs

This repository contains the code for our paper **MAGMA: Manifold Regularization for MAEs**.

MAGMA is a novel regularization technique that enhances self-supervised representation learning within masked autoencoders (MAE). The core of our approach lies in applying a manifold-based loss term that encourages consistency and smoothness between representations across different layers of the network. 

This codebase builds upon the [solo-learn repository](https://github.com/vturrisi/solo-learn).

## Abstract
Masked Autoencoders (MAEs) are an important divide in self-supervised learning (SSL) due to their independence from augmentation techniques for generating positive (and/or negative) pairs as in contrastive frameworks. Their masking and reconstruction strategy also nicely aligns with SSL approaches in natural language processing. Most MAEs are built upon Transformer-based architectures where visual features are not regularized as opposed to their convolutional neural network (CNN) based counterparts, which can potentially hinder their performance. To address this, we introduce \magma{}, a novel batch-wide layer-wise regularization loss applied to representations of different Transformer layers. We demonstrate that by plugging in the proposed regularization loss, one can significantly improve the performance of MAE-based models. We further demonstrate the impact of the proposed loss on optimizing other generic SSL approaches (such as VICReg and SimCLR), broadening the impact of the proposed approach.

## Installation
For installaing the environment follow the steps outlined in [solo-learn's README](solo/README.md)

## Usage
### Data Preparation

Prepare your datasets (e.g., ImageNet, CIFAR-100) following the instructions in the solo-learn repository.

### Training
To train an MAE model with MAGMA regularization, use the following command:

```bash
python main_pretrain.py --config-path scripts/pretrain/imagenet --config-name mae-reg-uniformity.yaml 
```

You can modify the [configuration file](scripts/pretrain/imagenet/mae-reg-uniformity.yaml) to adjust hyperparameters, dataset paths, and other settings.

Important parameters:
- `uniformity_weight`: Set to 0.01 to apply U-MAE's regularization. Set to 0 otherwise.
- `reg_scheduler`: Used to apply the `MAGMA` loss. 

## Citation
```(bibtex)
TODO
```

## Acknowledgements
This codebase builds upon the excellent work of the solo-learn repository. We thank the authors for their valuable contribution to the self-supervised learning community.

## License
This project is licensed under the MIT License.
