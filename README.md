# Conformal Bounds on Full-Reference Image Quality for Inverse Problems

## Description
This is the anonymized code for the paper "Conformal Bounds on Full-Reference Image Quality for Inverse Problems".

## Installation
Please follow the instructions to setup the environment to run the repo.
1. Create a new environment with the following commands
```
conda create -n qualityuq python=3.9 numpy=1.23 pip
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
conda install -c conda-forge cupy cudatoolkit=11.8 cudnn cutensor nccl
conda install -c anaconda h5py=3.6.0
```
2. From the project root directory, install the requirements with the following command
```
pip install -r requirements.txt
pip install qpsolvers[cvxopt,quadprog,highs]
```


# fastMRI Experiments

## Using Precomputed Metric Values
To run our conformal methods without needing to download the fastMRI datasets or the model weights, 
we include the precomputed FRIQ estimates and true FRIQs for each metric under results/VarNet/{metric}.
The FRIQ estiamtes are computed for a single E2EVarNet recovery compared against $c$ posterior samples from a CNF model.
To use these precomputed values, skip straight to the **Conformal Evaluation** section.
Otherwise, follow the instructions below to train a model and generate posterior samples.


## Usage Prerequisites
1. Download the fastMRI knee and brain datasets from [here](https://fastmri.org/)
2. Set the directory of the multicoil fastMRI knee and brain datasets to where they are stored on your device
    - Change [variables.py](variables.py) to set the paths to the dataset and your prefered logging folder
3. Change the configurations for training in the config files located in **train/configs/**. The current values are set to the ones used in the paper.


## Overview
- All model code used can be found in the **models** folder
- The training scripts can be found in the **train** folder
- The conformal prediction scripts can be found in the **conformal_metrics** folder
   - The underworkings of the conformal methods can be found in conformal__metrics_utils.py

## Training
First, set the directory to the **train** folder
```
cd train
```

To train a model, modify the configuration file in **train/configs/** and run the following commands for the model you want to train.
```
# Training a CNF model
python train_cnf.py --model_type MulticoilCNF 

# Training an E2E Model
python train_varnet.py
```

All models will be saved in the logging folder specified in [variables.py](variables.py)


## Conformal Evaluation

To perform the Monte Carlo evaluation, change the config file in **conformal_metrics/eval_config.py** to specify the location of the trained model (leave as default if using precomputed values)
, the type of conformal bound, the error rate, and all other parameters. Then, run the following commands
```
# Navigate to the conformal folder
cd conformal_metrics

# Run the evaluation
python eval_conformal_metrics.py 
```

To run the multi-round measurement protocol, run the following commands
```
# Navigate to the conformal folder
cd conformal_metrics

# Run the evaluation
python multi_round.py
```

Results will be saved in the results folder. 


# FFHQ Denoising
As with the MRI experiments, we include the precomputed FRIQ estimates and true FRIQs for each metric under results/ddrm/{metric}.
To use these precomputed values, skip straight to the **Conformal Evaluation** section.

Otherwise, follow the instructions below to generate posterior samples.
We utilize the code from the DDRM author's implementation found [here](https://github.com/bahjat-kawar/ddrm). This repo is stored in the folder labelled **ddrm**. Please follow the instructions from the DDRM repo to download the pretrained models and install the correct dependencies.


## Posterior Sample Generation
Since DDRM can take quite a long time to generate posterior samples, we first save  generated posterior samples for reuse. First, modify the file **ddrm/args/celeba_hq.yml** to specify the location of the FFHQ dataset and the desired logging location.

Then, run
```
cd ddrm

# Script to generate and save posterior samples
python generate_conformal_posteriors.py
```


## Conformal Evaluation
Similar to the MRI experiments, first edit the configuration file in **ddrm/eval_config.py** to specify desired parameters and the location of the saved images.

Then, run
```
eval_conformal_metrics_ddrm.py
```
This will save the results in the results folder as well.


## Notes
- The first time using a dataset will invoke the preprocessing step required for compressing the coils. 
Subsequent runs will be much faster since this step can be skipped.

