# Speech-Enhancement

## Introduction
This project provides an unofficial implementation of [SCP-GAN](https://arxiv.org/pdf/2210.14474.pdf), and its performance is compared with [CMGAN](https://arxiv.org/pdf/2209.11112.pdf) and [CDiffuSE](https://arxiv.org/pdf/2202.05256.pdf). There are some modifications made from their official implementations. 

## Getting Started
### Installation
This project was developed and tested in Ubuntu 22.04.
#### Step 1.
```shell
apt install g++
```
#### Step 2.
Install [PyTorch](https://pytorch.org/get-started/locally/). Follow their instructions. This project was tested on PyTorch version 2.0, and it also worked with version 1.13.
#### Step 3.
```shell
pip install -r requirements.txt
```

### Download VoiceBANK-DEMAND
This project used [VoiceBank-DEMAND](https://datashare.ed.ac.uk/handle/10283/2791) dataset. This dataset consists of 30 speakers from the VoiceBank corpus, which is further divided into a training set and a testing set with 28 and 2 speakers, respectively. The training set consists of 11,572 individual recordings of 28 speakers from the VoiceBank corpus mixed with DEMAND noises and some artificial background noises at the SNRs of 0, 5, 10, and 15. dB. The testing set has 824 utterances of 2 speakers mixed with unseen DEMAND noises at the SNRs of 2.5, 7.5, 12.5, and 17.5 dB. If you want to train/test yourself, please click "VoiceBank-DEMAND" above to download the dataset.

## Experiment results
## Discussion
## TODO

## References
- [Conditional Diffusion Probabilistic Model for Speech Enhancement](https://github.com/neillu23/CDiffuSE/tree/main)
- [CMGAN: Conformer-Based Metric GAN for Monaural Speech Enhancement](https://github.com/ruizhecao96/CMGAN)
- [Adaptive Weighted Discriminator for Training Generative Adversarial Networks](https://github.com/vasily789/adaptive-weighted-gans)
