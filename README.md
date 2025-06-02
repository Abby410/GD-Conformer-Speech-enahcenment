# GD-Conformer: Gated Dense Conformer for Speech Enhancement

## Overview
GD-Conformer is a state-of-the-art deep learning model designed for monaural speech enhancement. It integrates gated convolutional modules with a multi-stage Conformer architecture to effectively capture complex spectral and temporal dependencies in noisy speech signals. The model demonstrates superior performance on standard benchmarks such as VoiceBank+DEMAND and DNS Challenge 2020 datasets.

## Features
- Dual-path gated convolutional encoder modules for robust feature extraction
- Two-stage residual Conformer modules for spatio-temporal and frequency modeling
- Triple-domain loss function to optimize amplitude, phase, and complex spectral components
- Lightweight architecture with competitive computational complexity and parameter count

## Dataset
The model is trained and evaluated on:
- [VoiceBank+DEMAND](https://datashare.ed.ac.uk/handle/10283/2791)
- [DNS Challenge 2020](https://github.com/microsoft/DNS-Challenge)

## Installation
1. Clone this repository:
   ```bash
   git clone https://github.com/Abby410/GD-Conformer.git
   cd gd-conformer

  ## Install required packages:
  pip install -r requirements.txt
 ## Training
 Modify the configuration file config.yaml with your dataset paths and hyperparameters, then run:
 python train.py --config config.yaml

## Evaluation
After training, evaluate the model performance with:
python evaluate.py --model checkpoints/best_model.pth --dataset test_data/

## Results
GD-Conformer achieves state-of-the-art performance across multiple speech enhancement metrics including PESQ, STOI, CSIG, CBAK, and COVL.


## Contact
For questions or collaboration, please contact [1048456641@qq.com].
