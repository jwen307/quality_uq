
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
load_ckpt_dir = config['load_ckpt_dir']
model_folder = os.path.basename(os.path.normpath(load_ckpt_dir))

# Metrics to evaluate
metrics = ['dists', 'ssim', 'lpips', 'psnr']
num_ps = [1, 2, 4, 8, 16, 32]   # Number of posterior samples to average over
accels = [16, 8, 4, 2] # Accelerations to test
num_cs = [1,2,4,8,16,32] # Number of posterior samples to use for the regressor


# %% Get the arguments for training
if __name__ == "__main__":

    # Load the configuration for the CNF
    cnf_ckpt_name = 'best.ckpt'
    cnf_ckpt = os.path.join(load_ckpt_dir,
                            'checkpoints',
                            cnf_ckpt_name)

    # Get the configuration file for the CNF
    cnf_config_file = os.path.join(load_ckpt_dir, 'configs.pkl')
    cnf_config = helper.read_pickle(cnf_config_file)

    # Get the directory of the dataset
    base_dir = variables.fastmri_paths[cnf_config['data_args']['mri_type']]

    # Get the model type
    recon_type = load_ckpt_dir.split('/')[-3]

    if recon_type == 'VarNet':
        num_ps = [1]
        num_cs = [1]

    # Load the model
    recon_model = helper.load_model(recon_type, cnf_config, cnf_ckpt)


    recon_model.eval()
    recon_model.cuda()

    # Set up the conformal metric class for each metric
    cms = []
    recon_preds_dict = {}
    gt_preds_dict = {}
    save_dirs = []
    for metric in metrics:
        cms.append(conformal.ConformalMetrics(alpha, metric, device=recon_model.device,
                                              feature_preprocess_method=feature_preprocess_method,
                                              all_features=False, k = 5))
        # cm = conformal.ConformalMetrics(alpha, method, metric, loss, device=device)
        save_dir = os.path.join(load_ckpt_dir, 'conformal_metrics', metric, )
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


    #%% Compute posterior samples and metrics, then save the metrics to a file
    # Create the directory if it does not already exist and preprocess the files
    if not os.path.exists(os.path.join(save_dirs[0], 'recon_preds_dict.pkl')):

        for accel in accels:
            print('Acceleration: ', accel)


            data = FastMRIDataModule(base_dir,
                                     batch_size=cnf_config['train_args']['batch_size'],
                                     num_data_loader_workers=4,
                                     evaluating=True,
                                     specific_accel=accel,
                                     **cnf_config['data_args'],
                                     )
            data.prepare_data()
            data.setup()

            with torch.no_grad():
                print('Getting reconstructor predictions...')
                # TODO: Change back
                #for i in tqdm(range(len(data.val))):
                for i in tqdm(range(50)):

                    c, x, masks, norm_val, _,_,_ = data.val[i]
                    c = c.unsqueeze(0).to(recon_model.device)
                    x = x.to(recon_model.device)
                    if recon_type == 'VarNet':
                        masks = masks.unsqueeze(0).unsqueeze(1).to(recon_model.device)
                    else:
                        masks = masks.to(recon_model.device)
                    norm_val = norm_val.unsqueeze(0).to(recon_model.device)


                    # Get the reconstructions
                    samples = recon_model.reconstruct(c,
                                              num_samples=32+128,
                                              temp=1.0,
                                              check=True,
                                              maps=None,
                                              mask=masks,
                                              norm_val=norm_val,
                                              split_num=8,
                                              multicoil=False,
                                              rss=True)
                    recons = samples[0].to(recon_model.device)

                    # Get the ground truth
                    gt = fastmri.rss_complex(network_utils.format_multicoil(network_utils.unnormalize(x.unsqueeze(0), norm_val),
                                                                            chans=False), dim=1).to(recon_model.device)

                    for l, metric in enumerate(metrics):
                        # Get the distribution of metrics for the posteriors and the ground truth relative to the posterior mean
                        for p, num_p_avg in enumerate(num_ps):
                            if feature_preprocess_method == 'nonadaptive':
                                metrics_gt = cms[l].get_posterior_mean_metrics(recons.unsqueeze(0).unsqueeze(0), None, gt)
                                gt_preds_dict[metric][accel][num_p_avg].append(metrics_gt)
                                metrics_p = np.zeros((1, 128))
                                recon_preds_dict[metric][accel][num_p_avg].append(metrics_p)
                            else:
                                mean_p = recons[:num_p_avg].mean(dim=0).unsqueeze(0)
                                ps = recons[32:32+128]
                                metrics_p, metrics_gt = cms[l].get_posterior_mean_metrics(mean_p, ps, gt)

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



