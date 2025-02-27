
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
metrics = ['dists', 'ssim', 'lpips', 'psnr']
num_ps = [1, 2, 4, 8, 16, 32]   # Number of posterior samples to average over
accels = [16, 8, 4, 2] # Accelerations to test
num_cs = [32] #[1,2,4,8,16,32] # Number of posterior samples to use for the regressor


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




    #%% Test the conformal metrics
    print('Testing the conformal metrics...')


    for l, metric in enumerate(metrics):
        print('Metric: ', metric)
        xlim = None
        bins = None
        save_dir = save_dirs[l]

        mean_interval_size_dict = {}
        mean_worst_bound_dict = {}
        std_worst_bound_dict = {} #Standard error of the mean worst bound
        empirical_risk_std_dict = {} #Standard error of the empirical risk
        empirical_risk_dict = {}
        corr_dict = {}
        mean_conditional_risk_dict = {}
        pinball_losses_dict = {}
        pinball_losses_std_dict = {}

        for accel in accels:
            print('Acceleration: ', accel)

            mean_interval_size_dict[f'{accel}'] = {}
            mean_worst_bound_dict[f'{accel}'] = {}
            std_worst_bound_dict[f'{accel}'] = {}
            empirical_risk_std_dict[f'{accel}'] = {}
            empirical_risk_dict[f'{accel}'] = {}
            corr_dict[f'{accel}'] = {}
            mean_conditional_risk_dict[f'{accel}'] = {}
            pinball_losses_dict[f'{accel}'] = {}
            pinball_losses_std_dict[f'{accel}'] = {}

            accel_results_dir = os.path.join(save_dir, f'regressor_{feature_preprocess_method}', f'alpha{alpha}', f'accel{accel}')
            os.makedirs(accel_results_dir, exist_ok=True)

            for num_p_avg in num_ps:
                mean_interval_size_dict[f'{accel}'][f'{num_p_avg}'] = {}
                mean_worst_bound_dict[f'{accel}'][f'{num_p_avg}'] = {}
                std_worst_bound_dict[f'{accel}'][f'{num_p_avg}'] = {}
                empirical_risk_std_dict[f'{accel}'][f'{num_p_avg}'] = {}
                empirical_risk_dict[f'{accel}'][f'{num_p_avg}'] = {}
                corr_dict[f'{accel}'][f'{num_p_avg}'] = {}
                mean_conditional_risk_dict[f'{accel}'][f'{num_p_avg}'] = {}
                pinball_losses_dict[f'{accel}'][f'{num_p_avg}'] = {}
                pinball_losses_std_dict[f'{accel}'][f'{num_p_avg}'] = {}

                p_results_dir = os.path.join(accel_results_dir, f'num_p_avg{num_p_avg}')
                os.makedirs(p_results_dir, exist_ok=True)

                print('Num P Avg: ', num_p_avg)


                recon_preds = recon_preds_dict[metric][accel][num_p_avg]
                gt_preds = gt_preds_dict[metric][accel][num_p_avg]

                # For the non-adpative approach, simply set all recon_preds to be 0
                if feature_preprocess_method == 'nonadaptive':
                    recon_preds = np.zeros((recon_preds.shape[0], 1))

                # Make sure gt_preds are 1D
                gt_preds = gt_preds.flatten()



                #%% Get the empirical coverage
                for num_c in num_cs:
                    print('Num C: ', num_c)
                    c_results_dir = os.path.join(p_results_dir, f'num_c{num_c}')
                    os.makedirs(c_results_dir, exist_ok=True)

                    # Get the conformal metric object
                    cm = cms[l]

                    # Run the Monte Carlo trials
                    risks, mean_interval_sizes, mean_worst_bounds, corrs, bin_risks, pinball_losses = cm.eval_many_trials(recon_preds, gt_preds, num_trials, calib_split=calib_split, num_c=num_c, num_bins=5, correlation=True)

                    # Store the mean interval sizes
                    mean_interval_size_dict[f'{accel}'][f'{num_p_avg}'][f'{num_c}'] = np.mean(mean_interval_sizes)
                    mean_worst_bound_dict[f'{accel}'][f'{num_p_avg}'][f'{num_c}'] = np.mean(mean_worst_bounds)
                    std_worst_bound_dict[f'{accel}'][f'{num_p_avg}'][f'{num_c}'] = np.std(mean_worst_bounds)/np.sqrt(len(mean_worst_bounds))
                    empirical_risk_dict[f'{accel}'][f'{num_p_avg}'][f'{num_c}'] = np.mean(risks)
                    empirical_risk_std_dict[f'{accel}'][f'{num_p_avg}'][f'{num_c}'] = np.std(risks)/np.sqrt(len(risks))
                    corr_dict[f'{accel}'][f'{num_p_avg}'][f'{num_c}'] = np.mean(corrs)
                    mean_conditional_risk_dict[f'{accel}'][f'{num_p_avg}'][f'{num_c}'] = np.mean(bin_risks, axis=0).tolist()
                    pinball_losses_dict[f'{accel}'][f'{num_p_avg}'][f'{num_c}'] = np.mean(pinball_losses)
                    pinball_losses_std_dict[f'{accel}'][f'{num_p_avg}'][f'{num_c}'] = np.std(pinball_losses)/np.sqrt(len(pinball_losses))


                    with open(os.path.join(c_results_dir, f'empirical_risk_distrib'), 'w') as f:
                        f.write('Average Risk: {0:.5f} +/- {1:.5f}\n'.format(np.mean(risks), np.std(risks)/np.sqrt(len(risks))))
                        f.write('Average Interval Size: {0:.5f} +/- {1:.5f}\n'.format(np.mean(mean_interval_sizes), np.std(mean_interval_sizes)/np.sqrt(len(mean_interval_sizes))))



                    #%% Look at the scatter plot for a single trial
                    n = recon_preds.shape[0]

                    # Get the number of calibration samples
                    n_cal = int(n * calib_split)

                    # Get the indices for the calibration and test sets
                    indices = np.random.permutation(n)
                    cal_indices = indices[:n_cal]
                    test_indices = indices[n_cal:]

                    # Get the calibration and test sets
                    cal_recon_preds = recon_preds[cal_indices, :num_c]
                    cal_gt_preds = gt_preds[cal_indices]
                    test_recon_preds = recon_preds[test_indices, :num_c]
                    test_gt_preds = gt_preds[test_indices]

                    # Run for a single trial
                    risk, interval_size, worst_bounds, corr, bin_risks, pinball_loss = cm.eval_single_trial(cal_recon_preds,
                                                                                              cal_gt_preds,
                                                                                              test_recon_preds,
                                                                                              test_gt_preds,
                                                                                              correlation=True)


                    # Get a scatter plot of the true metric vs the worst case bound
                    plt.figure()
                    plt.rcParams.update({'font.size': 20})
                    plt.scatter(test_gt_preds, worst_bounds)
                    plt.axis('equal')
                    line_min = np.min([np.min(test_gt_preds), np.min(worst_bounds)])
                    line_max = np.max([np.max(test_gt_preds), np.max(worst_bounds)])
                    plt.plot([line_min, line_max], [line_min, line_max], 'r-')
                    plt.grid(alpha=0.5)
                    #plt.title('True Metric vs Interval Bound')
                    plt.savefig(os.path.join(c_results_dir, f'true_metric_vs_interval_bound.pdf'),
                                format='pdf', dpi=1200)
                    #plt.show()
                    plt.close()



        #%% Save the results
        file_save_dir = os.path.join(save_dir,f'regressor_{feature_preprocess_method}', f'alpha{alpha}')

        with open(os.path.join(file_save_dir, 'mean_interval_sizes.txt'), 'w') as f:
            f.write(json.dumps(mean_interval_size_dict))
        with open(os.path.join(file_save_dir, 'mean_worst_bounds.txt'), 'w') as f:
            f.write(json.dumps(mean_worst_bound_dict))
        with open(os.path.join(file_save_dir, 'std_worst_bounds.txt'), 'w') as f:
            f.write(json.dumps(std_worst_bound_dict))
        with open(os.path.join(file_save_dir, 'empirical_risk.txt'), 'w') as f:
            f.write(json.dumps(empirical_risk_dict))
        with open(os.path.join(file_save_dir, 'empirical_risk_std.txt'), 'w') as f:
            f.write(json.dumps(empirical_risk_std_dict))
        with open(os.path.join(file_save_dir, 'correlation.txt'), 'w') as f:
            f.write(json.dumps(corr_dict))
        with open(os.path.join(file_save_dir, 'mean_conditional_risk.txt'), 'w') as f:
            f.write(json.dumps(mean_conditional_risk_dict))
        with open(os.path.join(file_save_dir, 'pinball_losses.txt'), 'w') as f:
            f.write(json.dumps(pinball_losses_dict))
        with open(os.path.join(file_save_dir, 'pinball_losses_std.txt'), 'w') as f:
            f.write(json.dumps(pinball_losses_std_dict))



