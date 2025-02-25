import argparse
import traceback
import shutil
import logging
import yaml
import sys
import os
import torch
import numpy as np
import torch.utils.tensorboard as tb
from tqdm import tqdm

from runners.diffusion import Diffusion
from datasets import get_dataset, data_transform, inverse_data_transform
import torchvision.utils as tvu
import time
torch.set_printoptions(sci_mode=False)


def dict2namespace(config):
    namespace = argparse.Namespace()
    for key, value in config.items():
        if isinstance(value, dict):
            new_value = dict2namespace(value)
        else:
            new_value = value
        setattr(namespace, key, new_value)
    return namespace


config_path = 'configs/celeba_hq.yml'
args_path = 'args/celeba_hq.yml'

gpu_num=0

if __name__ == "__main__":

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    new_config = dict2namespace(config)

    with open(args_path, "r") as f:
        args_load = yaml.safe_load(f)
    args = dict2namespace(args_load)

    args.log_path = os.path.join(args.exp, "logs", args.doc)

    if not os.path.exists(os.path.join(args.exp, "image_samples")):
        os.makedirs(os.path.join(args.exp, "image_samples"), exist_ok=True)
    args.image_folder = os.path.join(args.exp, "image_samples", args.image_folder, str(args.sigma_0))
    if not os.path.exists(args.image_folder):
        os.makedirs(args.image_folder)

    # Create the image folders
    os.makedirs(os.path.join(args.image_folder, 'x'), exist_ok=True)
    os.makedirs(os.path.join(args.image_folder, 'y'), exist_ok=True)
    os.makedirs(os.path.join(args.image_folder, 'x_hat'), exist_ok=True)

    # add device
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    new_config.device = device

    try:
        # Get the model
        runner = Diffusion(args, new_config)
        runner.get_model()
        runner.get_Hfunc()
        #runner.sample()

        # Get the dataset
        data, _ = get_dataset(args, new_config)

        # Get a random subset of the dataset
        idx_file = os.path.join(args.image_folder, 'idxs.npy')
        if not os.path.exists(idx_file):
            idxs = np.random.choice(len(data), args.num_samples, replace=False)
            np.save(idx_file, idxs)
        else:
            idxs = np.load(idx_file)
        dataset = torch.utils.data.Subset(data, idxs)
        print(f'Dataset has size {len(dataset)}')
        #dataset = torch.utils.data.Subset(data, np.random.choice(len(data), args.num_samples, replace=False))

        # Divide the sample generation between the GPUs
        sample_split = len(dataset)

        times = []

        #for i in range(len(dataset)):
        for i in tqdm(range(gpu_num*sample_split, (gpu_num+1)*sample_split)):
            #Make the folder for the posterior samples
            os.makedirs(os.path.join(args.image_folder, 'x_hat', f"{i}"), exist_ok=True)

            # Get the image
            x_orig = dataset[i][0].unsqueeze(0).to(runner.device)
            classes = dataset[i][1]
            x_orig = data_transform(new_config, x_orig)

            # Get the measurement
            y, pinv_y = runner.get_measurements(x_orig)

            # Save the true image and measurement
            for j in range(len(pinv_y)):
                tvu.save_image(
                    inverse_data_transform(new_config, pinv_y[j]),
                    os.path.join(args.image_folder, 'y', f"{i}.png")
                )
                tvu.save_image(
                    inverse_data_transform(new_config, x_orig[j]),
                    os.path.join(args.image_folder, 'x', f"{i}.png")
                )

            # Generate several posterior samples
            for p in range(args.num_posteriors):
                start = time.time()
                x_hat = runner.get_posterior(y, classes)
                end = time.time()
                times.append(end-start)

                for j in range(x_hat[-1].size(0)):
                    tvu.save_image(
                        x_hat[-1][j], os.path.join(args.image_folder, 'x_hat', f"{i}", f"p{p}.png")
                    )

        print(f"Average time per sample: {np.mean(times)} +/- {np.std(times)}")


    except Exception:
        #logging.error(traceback.format_exc())
        print(traceback.format_exc())
