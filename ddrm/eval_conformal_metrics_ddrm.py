
import math
import numpy as np
import torch
import os
import fastmri
from tqdm import tqdm
import matplotlib.pyplot as plt
import json

import sys

sys.path.append("../")

from datasets.posteriors import PosteriorDataset
import conformal_metrics.conformal_metrics_utils as conformal
from ddrm.eval_config import Config

import pickle
import time


# Set the conformal parameters
config = Config().config
alpha = config['alpha']
feature_preprocess_method = config['feature_preprocess_method']
num_trials = config['num_trials']
calib_split = config['calib_split']
images_dir = config['images_dir']
num_train = config['num_train']
hyperparams = config['kfold_hyperparams']

metrics = ['dists', 'ssim', 'lpips', 'psnr']
num_ps = [1]
num_cs = [1,2,4,8,16,32]

if feature_preprocess_method == 'nonadaptive':
    num_cs = [1]


device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

# %% Get the arguments for training
if __name__ == "__main__":

    recon_preds_dict = {}
    gt_preds_dict = {}
    save_dirs = []

    cms = []
    for metric in metrics:
        cms.append(conformal.ConformalMetrics(alpha, metric, device=device,
                                              feature_preprocess_method=feature_preprocess_method,
                                              all_features=False, k=5))
        #cm = conformal.ConformalMetrics(alpha, method, metric, loss, device=device)
        save_dir = os.path.join(images_dir, 'conformal_metrics', metric)
        os.makedirs(save_dir, exist_ok=True)
        save_dirs.append(save_dir)
        recon_preds_dict[metric] = {}
        gt_preds_dict[metric] = {}


    # Create the directory if it does not already exist and preprocess the files
    if not os.path.exists(os.path.join(save_dirs[0], 'recon_preds_dict.pkl')):
        #os.makedirs(save_dir, exist_ok=True)

        data = PosteriorDataset(images_dir)

        for l, metric in enumerate(metrics):
            for p, num_p_avg in enumerate(num_ps):
                recon_preds_dict[metric][num_p_avg] = []
                gt_preds_dict[metric][num_p_avg] = []


        print('Getting classifier predictions...')
        # TODO: Switch back
        for i in tqdm(range(len(data))):
            c, x, recons = data[i]
            c = c.unsqueeze(0).to(device)
            x = x.to(device)
            recons = recons.to(device)


            for l, metric in enumerate(metrics):
                # Get the distribution of metrics for the posteriors and the ground truth relative to the posterior mean
                for p, num_p_avg in enumerate(num_ps):

                    mean_p = recons[:num_p_avg].mean(dim=0).unsqueeze(0)
                    ps = recons[32:32+128]
                    metrics_p, metrics_gt = cms[l].get_posterior_mean_metrics(mean_p, ps, x)

                    # Log the predictions
                    recon_preds_dict[metric][num_p_avg].append(metrics_p)
                    gt_preds_dict[metric][num_p_avg].append(metrics_gt)


        for l, metric in enumerate(metrics):
            for p in range(len(num_ps)):
                recon_preds_dict[metric][num_ps[p]] = np.stack(recon_preds_dict[metric][num_ps[p]], axis=0)
                gt_preds_dict[metric][num_ps[p]] = np.stack(gt_preds_dict[metric][num_ps[p]], axis=0)

        # Save the predictions
        for l, metric in enumerate(metrics):
            save_dir = save_dirs[l]
            with open(os.path.join(save_dir, 'recon_preds_dict.pkl'), 'wb') as f:
                pickle.dump(recon_preds_dict, f)
            with open(os.path.join(save_dir, 'gt_preds_dict.pkl'), 'wb') as f:
                pickle.dump(gt_preds_dict, f)


    else:
        print('Loading predictions...')
        # recon_preds = np.load(os.path.join(save_dir, 'recon_preds.npy'))
        # gt_preds = np.load(os.path.join(save_dir, 'gt_preds.npy'))
        for l, metric in enumerate(metrics):
            save_dir = save_dirs[l]
            with open(os.path.join(save_dir, 'recon_preds_dict.pkl'), 'rb') as f:
                recon_preds_dict = pickle.load(f)
            with open(os.path.join(save_dir, 'gt_preds_dict.pkl'), 'rb') as f:
                gt_preds_dict = pickle.load(f)








    #%% Test the conformal metrics
    print('Testing the conformal metrics...')
    mean_interval_size_dict = {}
    mean_worst_bound_dict = {}
    std_worst_bound_dict = {}  # Standard error of the mean worst bound
    mean_conditional_risk_dict = {}
    mean_corr_dict = {}
    pinball_losses_dict = {}
    pinball_losses_std_dict = {}

    np.random.seed(0)
    indices = np.random.permutation(recon_preds_dict['dists'][1].shape[0])
    train_indices = indices[:num_train]
    cal_test_indices = indices[num_train:]



    for l, metric in enumerate(metrics):
        xlim = None
        bins = None

        save_dir = save_dirs[l]


        #for num_p_avg in num_ps:
        for num_c in num_cs:

            results_dir = os.path.join(save_dir, f'regressor_{feature_preprocess_method}', f'alpha{alpha}', f'num_c{num_c}')

            os.makedirs(results_dir, exist_ok=True)

            print('Num C: ', num_c)

            recon_preds = recon_preds_dict[metric][1]
            gt_preds = gt_preds_dict[metric][1]

            cm = cms[l]

            # Train the preprocess model
            train_recon_preds = recon_preds[:num_train,:num_c]
            train_gt_preds = gt_preds[:num_train]

            recon_preds = recon_preds[num_train:]
            gt_preds = gt_preds[num_train:]

            # Train the regressor
            cm.fit_preprocessor(train_recon_preds, train_gt_preds, hyperparams=hyperparams)


            # Evaluate over Monte Carlo trials
            risks, mean_interval_sizes, mean_worst_bounds, corr, bin_risks, pinball_losses = cm.eval_many_trials(recon_preds, gt_preds, num_trials, calib_split=calib_split, correlation=True, num_bins=5, num_c=num_c)

            # Store the mean interval sizes
            mean_interval_size_dict[f'{num_c}'] = np.mean(mean_interval_sizes).astype(np.double)
            mean_worst_bound_dict[f'{num_c}'] = np.mean(mean_worst_bounds).astype(np.double)
            std_worst_bound_dict[f'{num_c}'] = np.std(mean_worst_bounds).astype(np.double) / np.sqrt(len(mean_worst_bounds)).astype(np.double)
            mean_conditional_risk_dict[f'{num_c}'] = np.mean(bin_risks, axis=0).astype(np.double).tolist()
            mean_corr_dict[f'{num_c}'] = np.mean(corr).astype(np.double)
            pinball_losses_dict[f'{num_c}'] = np.mean(pinball_losses).astype(np.double)
            pinball_losses_std_dict[f'{num_c}'] = np.std(pinball_losses).astype(np.double) / np.sqrt(len(pinball_losses)).astype(np.double)


            with open(os.path.join(results_dir, f'empirical_risk_distrib'), 'w') as f:
                f.write('Average Risk: {0:.5f} +/- {1:.5f}\n'.format(np.mean(risks), np.std(risks)/np.sqrt(len(risks))))
                f.write('Average Interval Size: {0:.5f} +/- {1:.5f}\n'.format(np.mean(mean_interval_sizes), np.std(mean_interval_sizes)/np.sqrt(len(mean_interval_sizes))))


            #%% Look at a scatter plot for a single trial
            n = recon_preds.shape[0]

            # Get the number of calibration samples
            n_cal = int(n * calib_split)

            # Get the indices for the calibration and test sets
            #indices = np.random.permutation(n)
            indices = np.arange(n) # Keeps the indices the same for each trial
            cal_indices = indices[:n_cal]
            test_indices = indices[n_cal:]

            # Get the calibration and test sets
            cal_recon_preds = recon_preds[cal_indices, :num_c]
            cal_gt_preds = gt_preds[cal_indices]
            test_recon_preds = recon_preds[test_indices, :num_c]
            test_gt_preds = gt_preds[test_indices]

            # Run for a single trial
            risk, interval_size, worst_bounds, corr, bin_risks, pinball_losses = cm.eval_single_trial(cal_recon_preds, cal_gt_preds, test_recon_preds, test_gt_preds, correlation=True)



            # Get a scatter plot of the true metric vs the worst case bound
            plt.figure()
            plt.scatter(test_gt_preds, worst_bounds)
            plt.axis('equal')
            line_min = np.min([np.min(test_gt_preds), np.min(worst_bounds)])
            line_max = np.max([np.max(test_gt_preds), np.max(worst_bounds)])
            plt.plot([line_min, line_max], [line_min, line_max], 'r-')
            plt.savefig(os.path.join(results_dir, f'true_metric_vs_interval_bound.pdf'),
                        format='pdf', dpi=1200)
            # plt.show()
            plt.close()



        #%% Save and plot the mean interval sizes
        file_save_dir = os.path.join(save_dir, f'regressor_{feature_preprocess_method}', f'alpha{alpha}')
        with open(os.path.join(file_save_dir, 'mean_interval_sizes.txt'), 'w') as f:
            f.write(json.dumps(mean_interval_size_dict))
        with open(os.path.join(file_save_dir, 'mean_worst_bounds.txt'), 'w') as f:
            f.write(json.dumps(mean_worst_bound_dict))
        with open(os.path.join(file_save_dir, 'std_worst_bounds.txt'), 'w') as f:
            f.write(json.dumps(std_worst_bound_dict))
        with open(os.path.join(file_save_dir, 'mean_conditional_risks.txt'), 'w') as f:
            f.write(json.dumps(mean_conditional_risk_dict))
        with open(os.path.join(file_save_dir, 'mean_correlations.txt'), 'w') as f:
            f.write(json.dumps(mean_corr_dict))
        with open(os.path.join(file_save_dir, 'pinball_losses.txt'), 'w') as f:
            f.write(json.dumps(pinball_losses_dict))
        with open(os.path.join(file_save_dir, 'pinball_losses_std.txt'), 'w') as f:
            f.write(json.dumps(pinball_losses_std_dict))


