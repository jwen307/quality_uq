
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
from conformal_metrics.eval_config import Config

from datasets.fastmri_multicoil import FastMRIDataModule
from util import helper
import variables
import conformal_metrics_utils as conformal
from util import network_utils
import pickle


# Get the configurations
config = Config().config
alpha = config['alpha']
feature_preprocess_method = config['feature_preprocess_method']
num_trials = config['num_trials']
calib_split = config['calib_split']

# Get the checkpoints
load_recovery_ckpt = config['load_recovery_ckpt']
load_posterior_ckpt = config['load_posterior_ckpt']
recovery_model_folder = os.path.basename(os.path.normpath(load_recovery_ckpt))
posterior_model_folder = os.path.basename(os.path.dirname(load_posterior_ckpt))

# Metrics to evaluate
metrics = ['dists'] #['dists', 'ssim', 'lpips', 'psnr']
num_ps = [1, 2, 4, 8, 16, 32]   # Number of posterior samples to average over
accels = [16, 8, 4, 2] # Accelerations to test
num_cs = [32] #[1,2,4,8,16,32] # Number of posterior samples to use for the regressor


# Parameters for this run of the multiround procedure
num_p = 1 # Number of posterior samples to average over
num_c = 32 # Number of samples to compare against
tau = 0.16  # Threshold for the metric bound


def get_model_from_ckpt(ckpt_dir):
    """
    Get the model from the checkpoint
    """

    # Load the configureation file for the mdoel
    ckpt_name = 'best.ckpt'
    ckpt = os.path.join(ckpt_dir, 'checkpoints', ckpt_name)

    # Get the configuration file for the model
    config_file = os.path.join(ckpt_dir, 'configs.pkl')
    config = helper.read_pickle(config_file)

    # Get the model type
    model_type = ckpt_dir.split('/')[-3]

    # Load the model
    print('Loading {model_type}'.format(model_type=model_type))
    model = helper.load_model(model_type, config, ckpt)

    model.eval()
    model.cuda()

    return model, model_type, config


# %% Get the arguments for training
if __name__ == "__main__":

    # Get the recovery type
    recovery_type = load_recovery_ckpt.split('/')[-3]

    # Set up the conformal metric class for each metric
    cms = []
    recon_preds_dict = {}
    gt_preds_dict = {}
    save_dirs = []
    for metric in metrics:
        cms.append(conformal.ConformalMetrics(alpha, metric, device='cuda',
                                              feature_preprocess_method=feature_preprocess_method,
                                              all_features=False, k=5))
        # cm = conformal.ConformalMetrics(alpha, method, metric, loss, device=device)
        # save_dir = os.path.join(load_recovery_ckpt, 'conformal_metrics', metric, )
        proj_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        save_dir = os.path.join(proj_dir, 'results', recovery_type, metric)
        os.makedirs(save_dir, exist_ok=True)
        save_dirs.append(save_dir)
        recon_preds_dict[metric] = {}
        gt_preds_dict[metric] = {}

        for accel in accels:
            recon_preds_dict[metric][accel] = {}
            gt_preds_dict[metric][accel] = {}
            for num_p_avg in num_ps:
                recon_preds_dict[metric][accel][num_p_avg] = []
                gt_preds_dict[metric][accel][num_p_avg] = []

    # Set p to be 1 for VarNet since only generates a single sample
    if recovery_type == 'VarNet':
        num_ps = [1]



    #%% Compute posterior samples and metrics, then save the metrics to a file
    # Create the directory if it does not already exist and preprocess the files
    if not os.path.exists(os.path.join(save_dirs[0], 'recon_preds_dict.pkl')):

        # Get the recovery model
        recovery_model, recovery_type, recovery_config = get_model_from_ckpt(load_recovery_ckpt)

        # Get the posterior model
        posterior_model, posterior_type, posterior_config = get_model_from_ckpt(load_posterior_ckpt)

        # Get the directory of the dataset
        base_dir = variables.fastmri_paths[posterior_config['data_args']['mri_type']]

        for accel in accels:
            print('Acceleration: ', accel)


            data = FastMRIDataModule(base_dir,
                                     batch_size=recovery_config['train_args']['batch_size'],
                                     num_data_loader_workers=4,
                                     evaluating=True,
                                     specific_accel=accel,
                                     **recovery_config['data_args'],
                                     )
            data.prepare_data()
            data.setup()

            with torch.no_grad():
                print('Getting reconstructor predictions...')
                for i in tqdm(range(len(data.val))):
                #for i in tqdm(range(50)):

                    c, x, masks, norm_val, _,_,_ = data.val[i]
                    c = c.unsqueeze(0).to(recovery_model.device)
                    x = x.to(recovery_model.device)

                    # VarNet needs the mask in a different format
                    if recovery_type == 'VarNet':
                        recovery_masks = masks.unsqueeze(0).unsqueeze(1).to(recovery_model.device)
                    else:
                        recovery_masks = masks.to(recovery_model.device)

                    posterior_masks = masks.to(posterior_model.device)
                    norm_val = norm_val.unsqueeze(0).to(recovery_model.device)


                    # Get the reconstructions
                    recons = recovery_model.reconstruct(c,
                                              num_samples=max(num_ps),
                                              temp=1.0,
                                              check=True,
                                              maps=None,
                                              mask=recovery_masks,
                                              norm_val=norm_val,
                                              split_num=8,
                                              multicoil=False,
                                              rss=True)
                    if recovery_type == 'VarNet':
                        recons = recons[0].to(recovery_masks.device).unsqueeze(0).unsqueeze(0)
                    else:
                        recons = recons[0].to(recovery_masks.device)

                    # Get the posteriors
                    posteriors = posterior_model.reconstruct(c,
                                                         num_samples=32,
                                                         temp=1.0,
                                                         check=True,
                                                         maps=None,
                                                         mask=posterior_masks,
                                                         norm_val=norm_val,
                                                         split_num=8,
                                                         multicoil=False,
                                                         rss=True)
                    posteriors = posteriors[0].to(recovery_masks.device)

                    # Get the ground truth
                    gt = fastmri.rss_complex(network_utils.format_multicoil(network_utils.unnormalize(x.unsqueeze(0), norm_val),
                                                                            chans=False), dim=1).to(recovery_model.device)

                    for l, metric in enumerate(metrics):
                        # Get the distribution of metrics for the posteriors and the ground truth relative to the posterior mean
                        for p, num_p_avg in enumerate(num_ps):

                            mean_p = recons[:num_p_avg].mean(dim=0).unsqueeze(0)
                            metrics_p, metrics_gt = cms[l].get_posterior_mean_metrics(mean_p, posteriors, gt)

                            # Log the predictions
                            recon_preds_dict[metric][accel][num_p_avg].append(metrics_p)
                            gt_preds_dict[metric][accel][num_p_avg].append(metrics_gt)

            for l, metric in enumerate(metrics):
                for num_p_avg in num_ps:
                    recon_preds_dict[metric][accel][num_p_avg] = np.stack(recon_preds_dict[metric][accel][num_p_avg], axis=0)
                    gt_preds_dict[metric][accel][num_p_avg] = np.stack(gt_preds_dict[metric][accel][num_p_avg], axis=0)


        # Save the predictions
        for l, metric in enumerate(metrics):
            save_dir = save_dirs[l]
            with open(os.path.join(save_dir, 'recon_preds_dict.pkl'), 'wb') as f:
                pickle.dump(recon_preds_dict, f)
            with open(os.path.join(save_dir, 'gt_preds_dict.pkl'), 'wb') as f:
                pickle.dump(gt_preds_dict, f)


    # Just load the predictions if already computed
    else:
        print('Loading predictions...')
        for l, metric in enumerate(metrics):
            save_dir = save_dirs[l]
            with open(os.path.join(save_dir, 'recon_preds_dict.pkl'), 'rb') as f:
                recon_preds_dict = pickle.load(f)
            with open(os.path.join(save_dir, 'gt_preds_dict.pkl'), 'rb') as f:
                gt_preds_dict = pickle.load(f)




    #%% Test the multiround protocol
    print('Testing the multiround procedure...')

    all_coverages = []
    all_avg_accels = []
    all_num_accepted = []

    for t in tqdm(range(num_trials)):
        # Get the calibration and test splits
        recon_preds = recon_preds_dict[metric][16][1]
        # Total number of samples
        n = recon_preds.shape[0]
        # Get the number of calibration samples
        n_cal = int(n * calib_split)
        # Get the indices for the calibration and test sets (Keep same for all accelerations)
        indices = np.random.permutation(n)
        cal_indices = indices[:n_cal]
        test_indices = indices[n_cal:]

        total_test = n - n_cal

        metric = metrics[0]
        l = 0
        xlim = None
        bins = None
        save_dir = os.path.join(save_dirs[l], 'multiround')

        num_accepted = [0 for _ in range(len(accels) + 1)]  # Number of samples accepted for each acceleration
        accepted_coverages = [0 for _ in range(len(accels))]
        slice_accels = []
        num_in_interval = 0

        for i, accel in enumerate(accels):

            accel_results_dir = os.path.join(save_dir, f'regressor_{feature_preprocess_method}',
                                             f'alpha{alpha}', f'tau{tau}')
            os.makedirs(accel_results_dir, exist_ok=True)

            p_results_dir = os.path.join(accel_results_dir, f'num_p_avg{num_p}')
            os.makedirs(p_results_dir, exist_ok=True)

            recon_preds = recon_preds_dict[metric][accel][num_p]
            gt_preds = gt_preds_dict[metric][accel][num_p]

            # train_recon_preds = recon_preds_training_dict[metric][accel][num_p]
            # train_gt_preds = gt_preds_training_dict[metric][accel][num_p]

            # For case when you want to just use the CNF for the non-adaptive approach
            if feature_preprocess_method == 'nonadaptive':
                recon_preds = np.zeros((recon_preds.shape[0], 1))
                # train_recon_preds = np.zeros((train_recon_preds.shape[0], 1))

            c_results_dir = os.path.join(p_results_dir, f'num_c{num_c}')
            os.makedirs(c_results_dir, exist_ok=True)

            # Get the training samples for the regressor
            # train_recon_preds_c = train_recon_preds[:,:num_c]
            cm = cms[l]
            # cm.fit_preprocessor(train_recon_preds_c, train_gt_preds, hyperparams=hyperparams)

            # Get the calibration and test sets
            cal_recon_preds = recon_preds[cal_indices, :num_c]
            cal_gt_preds = gt_preds[cal_indices].flatten()
            test_recon_preds = recon_preds[test_indices, :num_c]
            test_gt_preds = gt_preds[test_indices]

            # Calibrate the conformal metrics
            cm.compute_lambda(cal_recon_preds, cal_gt_preds)

            # Get the bounds for the test set
            intervals, _ = cm.conformal_inference(test_recon_preds)

            # Get the bound on the interval and see if it satifies the tau threshold
            if cm.metric == 'psnr' or cm.metric == 'ssim':
                bound = intervals[:, 0]
                # See which samples are above the threshold
                accepted = bound >= tau

            else:
                bound = intervals[:, 1]
                # See which samples are below the threshold
                accepted = bound <= tau

            # Get the number of samples that satisfy the threshold
            num_accepted[i] = np.sum(accepted)

            if any(accepted):

                risk = cm.get_risk(intervals[accepted], test_gt_preds[accepted])

                # Get the empirical coverage for the accepted samples
                accepted_coverages[i] = 1 - risk

                # Go through all of the accepted samples
                accepted_intervals = intervals[accepted]
                accepted_gt_preds = test_gt_preds[accepted]
                for j in range(len(accepted_intervals)):
                    if accepted_intervals[j, 0] <= accepted_gt_preds[j] and accepted_intervals[j, 1] >= accepted_gt_preds[j]:
                        num_in_interval += 1

                    # Add the slice acceleration to the list
                    slice_accels.append(accel)

            # Only keep the test samples that were not accepted for the next acceleration
            # test_recon_preds = test_recon_preds[~accepted]
            # test_gt_preds = test_gt_preds[~accepted]
            test_indices = test_indices[~accepted]

            # Break when all samples have been accepted
            if test_indices.shape[0] == 0:
                break

        # Include the fully sampled images if threshold never went below tau
        if test_indices.shape[0] > 0:
            num_in_interval += len(test_indices)

            slice_accels += [1 for _ in range(len(test_indices))]

            num_accepted[-1] = len(test_indices)

        # Make sure that the number of samples accepted is the same as the total number of samples
        assert sum(num_accepted) == total_test

        # Compute
        # Empirical Coverage when each slice was accepted
        coverage = num_in_interval / total_test
        # print('Coverage: ', coverage)
        all_coverages.append(coverage)

        # Get the average acceleration
        avg_accel = 1 / np.mean(1 / np.array(slice_accels))
        # print('Average Acceleration: ', avg_accel)
        all_avg_accels.append(avg_accel)

        # Get the number of samples accepted at each acceleration rate
        all_num_accepted.append(num_accepted)

    # %% Get the average coverage and average acceleration across trials with standard error
    print(f'Average Coverage: {np.mean(all_coverages)} +/- {np.std(all_coverages) / np.sqrt(num_trials)}')
    print(f'Average Acceleration: {np.mean(all_avg_accels)} +/- {np.std(all_avg_accels) / np.sqrt(num_trials)}')

    # Get the average number of samples accepted at each acceleration rate
    all_percent_accepted = np.array(all_num_accepted) / total_test
    mean_percent_accepted = np.mean(all_percent_accepted, axis=0)
    std_error_percent_accepted = np.std(all_percent_accepted, axis=0) / np.sqrt(num_trials)
    std_dev_percent_accepted = np.std(all_percent_accepted, axis=0)
    print('Average Number of Samples Accepted: ', mean_percent_accepted)
    print('Standard Error of Number of Samples Accepted: ', std_error_percent_accepted)
    print('Standard Deviation of Number of Samples Accepted: ', std_dev_percent_accepted)

    # %% Save the results
    with open(os.path.join(c_results_dir, 'avg_coverage.txt'), 'w') as f:
        f.write(f'Average Coverage: {np.mean(all_coverages)} +/- {np.std(all_coverages) / np.sqrt(num_trials)}')
    with open(os.path.join(c_results_dir, 'avg_accel.txt'), 'w') as f:
        f.write(f'Average Acceleration: {np.mean(all_avg_accels)} +/- {np.std(all_avg_accels) / np.sqrt(num_trials)}')
    with open(os.path.join(c_results_dir, 'avg_num_accepted.pkl'), 'wb') as f:
        pickle.dump({'mean': mean_percent_accepted, 'std_error': std_error_percent_accepted,
                     'std_dev': std_dev_percent_accepted}, f)


    #%% Save plot showing the number of samples accepted at each acceleration rate

    plt.figure()
    # Define the bard width
    bar_width = 0.35
    x = np.arange(1, 6)
    r1 = x - bar_width / 2
    r2 = x + bar_width / 2

    # Plot the bars with error bars
    plt.bar(r1, mean_percent_accepted, bar_width, yerr=std_dev_percent_accepted)

    # plt.legend()
    plt.gca().set_xticks(x)
    plt.gca().set_xticklabels([16, 8, 4, 2, 1])
    # plt.show()
    plt.savefig(os.path.join(c_results_dir, f'accepted_accelerations_tau{tau}_cnf.pdf'), format='pdf', dpi=1200)




