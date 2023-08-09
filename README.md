# Dual-input neural networks (DI-NNs)
This repository contains the code for the neural networks and training scripts
related to the paper "Dual input neural networks for positional sound source localization"
by Eric Grinstein, Vincent W. Neo, Patrick A. Naylor

Link: https://arxiv.org/abs/2308.04169

## Overview
* The `di_nn` repository contains the networks and training of our neural networks.
* The file `di_nn/utils/di_crnn.py` contains a base Torch model that is unrelated to our application of sound source localization. It is therefore the recommended entry point for those who want to use DI-NNs on their own domains.
* In turn, the file `di_nn/di_ssl_net.py` contains the network adapted for the task of sound source localization.
* The file `di_nn/trainer.py` contains a Pytorch lightning model for training.
* The [pysoundloc](https://github.com/SOUNDS-RESEARCH/pysoundloc) submodule contains the Least Squares baseline. 
* The [sydra](https://github.com/SOUNDS-RESEARCH/pysoundloc) submodule contains the code for generating the synthetic datasets used in the paper. It is included here for convenience, but is a separate project available [here](https://github.com/SOUNDS-RESEARCH/sydra)


## Installation

Clone the repository using the following command:
`git clone https://github.com/egrinstein/di_nn --recurse-submodules 

The requirements of this project are listed in the file `requirements.txt`
Use the command `pip install -r requirements.txt` to install them.

You can also train the model using the Kaggle notebook available [here](https://www.kaggle.com/code/egrinstein/di-nn-training-notebook).
Note you'll need a phone-verified Kaggle account to use the GPU.

## Testing the model
Under the directory `demo/`, you will find a Jupyter notebook as well as the model's pretrained weights and a small testing dataset.  

## Generating the datasets
Synthetic data was generated using a package created by the authors called SYDRA (SYnthetic Datasets for Room Acoustics).
This package is included here for convenience under the `sydra` directory. The configuration of each generated dataset is governed by [Hydra](www.hydra.cc).


### Synthetic datasets
To generate a synthetic dataset, one must change the configuration under `sydra/config/config.yaml` to generate the desired synthetic dataset.
Then, generate a dataset by running the command: `python main.py dataset_dir=path/to/dataset num_samples=X`.
after modifying

### Recorded datasets
To generate a dataset using the [LibriAdhoc40](https://github.com/ISmallFish/Libri-adhoc40) recorded dataset, you must first download it.
Then, change directory to `sydra/adhoc40_dataset` and run the command `python generate_dataset.py input_dir=/path/to/libri_adhoc40_dataset output_dir=/output/path mode='train|validation|test'`
to generate the training, validation or testing datasets. You can alternatively alter the configuration under `sydra/config/adhoc40_dataset.yaml`

## Training
Training, the datasets and model are also configured using Hydra. You can alter these configs at `di_nn/config`.
Once the datasets are available, you can train the models by running `python train.py`.

## Evaluating the Least Squares Sound Source Localization (LS-SSL) baseline
The code for the baseline is located under the `pysoundloc/` directory. To run the tests, run `python test_ls_ssl_baseline.py`. The choice of which dataset to evaluate the baseline on is govern by the same .yaml files above