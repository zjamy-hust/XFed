# XFed: Improving Explainability in Federated Learning by Intersection Over Union Ratio Extended Client Selection
This repository contains the code for our ECAI'23 paper "XFed: Improving Explainability in Federated Learning by Intersection Over Union Ratio Extended Client Selection".

## Quick Links

- [Overview](#overview)
- [Run Code](#run-code)
  - [Requirements](#requirements)
  - [Training](#training)
- [Bugs or Questions?](#bugs-or-questions)
<!-- - [Citation](#citation) -->
<!-- - [Acknowledgment](#acknowledgment) -->


## Overview
Federated Learning (FL) allows massive clients to collaboratively train a global model without revealing their private data. 
Because of the participants' not independently and identically distributed (non-IID) statistical characteristics, it will cause divergence among the client’s Deep Neural Network model weights and require more communication rounds before training can be converged. Moreover, models trained from non-IID data may also extract biased features and the rationale behind the model is still not fully analyzed and exploited.
In this paper, we propose eXplainable-Fed (XFed) which is a novel client selection mechanism that takes both accuracy and explainability into account.
Specifically, XFed selects participants in each round based on a small test set's accuracy via cross-entropy loss and interpretability via XAI-accuracy. 
XAI-accuracy is calculated by Intersection over Union Ratio between the heat map and the truth mask to evaluate the overall rationale of accuracy. 
The results of our experiments show that our method has comparable accuracy to state-of-the-art methods specially designed for accuracy while increasing explainability by 14\%-35\% in terms of rationality.


## Run Code
### Requirements

To install the dependencies, please run
```
pip install -r requirements.txt
```

### Training

run_mode_1.sh displays the parameters related to running the code.
```
bash run_mode_1.sh
```

fedxai.py is the main entrance.

xai.py mainly focuses on xai operations.

assets includes the truth masks.

## Bugs or Questions

If you have any questions about our paper, please reach out to Ruixuan Li (rxli@hust.edu.cn). Also, feel free to open an issue for bug reporting.


