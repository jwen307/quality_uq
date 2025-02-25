import torch
import numpy as np
from DISTS_pytorch import DISTS
import itertools

from matplotlib import pyplot as plt
from tqdm import tqdm
from scipy.optimize import brentq, linprog
from sklearn.linear_model import QuantileRegressor, LinearRegression
from sklearn.preprocessing import RobustScaler, QuantileTransformer
from torchmetrics.image import StructuralSimilarityIndexMeasure, PeakSignalNoiseRatio
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
from qpsolvers import solve_qp
from xgboost import XGBRegressor


# Class to handle the conformal procedure
class ConformalMetrics:
    def __init__(self, alpha=0.1,  metric='ssim', loss='single_indicator', device='cuda', feature_preprocess_method='quantile', all_features=False, k=5):
        '''
        alpha: error level
        metric: metric to use
        feature_preprocess_method: method to preprocess the features ('nonadaptive', 'quantile', 'regression')
        '''


        if metric == 'dists':
            self.D = DISTS().to(device)
        elif metric == 'ssim':
            self.D = StructuralSimilarityIndexMeasure(reduction=None).to(device)
        elif metric == 'psnr':
            #self.D = PeakSignalNoiseRatio(reduction=None).to(device)
            self.D = PSNR(device=device)
        elif metric == 'lpips':
            self.D = LearnedPerceptualImagePatchSimilarity(net_type='vgg',normalize=True).to(device)

        self.metric = metric
        self.loss = loss
        self.alpha = alpha

        self.lambda_hat = None

        if loss=='single_indicator':
            self.B=1 #Upper bound on loss


        # Processes the features before conformalizing
        if metric == 'dists' or metric == 'lpips':
            alpha_process = 1-self.alpha
        else:
            alpha_process = self.alpha
        self.feature_preprocessor = FeaturePreprocessor(alpha_process, feature_preprocess_method, all_features=all_features,device=device, k=k)



    def get_posterior_mean_metrics(self, mean, recons, gt):
        """
        Get the metric for the reconstructions pairwise
        :param recons: (p, 1, h, w) tensor of reconstructions
        :param metric:
        :return: pairwise metrics for the posterior samples
        """


        if self.feature_preprocessor.method != 'nonadaptive':
            # Get the number of posterior samples
            p = recons.shape[0]

            # Repeat the mean of the posterior samples
            mean_recon = mean.repeat(p,1,1,1)

            # For some reason, LPIPS and PSNR either sums or averages in the batch dimension so do separately
            if self.metric == 'lpips': # or self.metric == 'psnr':
                metric_p = []
                for i in range(p):
                    metric_p.append(self.compute_metric(mean_recon[i].unsqueeze(0), recons[i].unsqueeze(0)))
                metric_p = np.stack(metric_p, axis=0)
            else:
                metric_p = self.compute_metric(mean_recon, recons)

        # Get the metric of the mean relative to the gt
        #Repeat the ground truth to match the number of samples
        if len(gt.shape) < 4:
            gt = gt.unsqueeze(0)
        metric_gt = self.compute_metric(mean, gt)

        if self.feature_preprocessor.method != 'nonadaptive':
            return metric_p, metric_gt
        else:
            return  metric_gt


    def compute_metric(self, recons, gt):
        # For FFHQ experiments where already in RGB format
        if recons.shape[1] == 3:
            return self.compute_single_channel_metric(recons, gt)

        # For the MRI experiments
        else:
            if self.metric == 'dists' or self.metric == 'lpips':
                return self.compute_rgb_metric(recons, gt)
            elif self.metric == 'ssim' or self.metric == 'psnr':
                return self.compute_single_channel_metric(recons, gt)
            else:
                raise ValueError('Metric not implemented')

    def compute_rgb_metric(self, recon, gt):
        """
        Compute the DISTS or LPIPS metric between the reconstruction and the ground truth
        :param recon: (b, 1, h, w) tensor of the reconstruction
        :param gt: (b, 1, h, w) tensor of the ground truth
        :return: DISTS metric
        """

        # Repeat the channel dimensions
        recon = to_rgb(recon)
        gt = to_rgb(gt)

        # Normalize to be between 0 and 1
        recon = normalize(recon)
        gt = normalize(gt)

        return self.D(recon, gt).detach().cpu().numpy()

    def compute_single_channel_metric(self, recon, gt):
        """
        Compute the SSIM or PSNR metric between the reconstruction and the ground truth
        :param recon: (b, 1, h, w) tensor of the reconstruction
        :param gt: (b, 1, h, w) tensor of the ground truth
        :return: SSIM or PSNR metric
        """

        return self.D(recon, gt).detach().cpu().numpy()


    def get_interval(self, recon_preds, lam):
        """
        Get the conformal interval
        :param recon_preds: (n, m) array of metrics for the reconstructions
        :param lam: lambda value to control the width of the interval
        :param alpha: error level
        :return: (m, 2) array of the conformal interval
        """

        # Get the processed features
        recon_preds = self.feature_preprocessor.predict(recon_preds)

        # Get the quantiles
        if self.metric == 'psnr' or self.metric == 'ssim':
            lower = -lam + recon_preds
            upper = np.ones_like(lower) * 50 if self.metric == 'psnr' else np.ones_like(
                lower)  # Set to a very high value

        else:
            upper = lam + recon_preds
            lower = np.zeros_like(upper)

        interval_size = upper - lower

        return np.stack([lower, upper], axis=-1), interval_size


    def get_risk(self, intervals, gt_preds):
        """
        Get the loss for the conformal interval
        :param intervals: (n, 2) array of intervals
        :param gt_preds: (n, p) array of ground truth metrics
        :return:
        """

        return single_indicator_risk(intervals, gt_preds)


    def get_bin_risk(self, intervals, gt_preds, num_bins=10):
        # Get the risk for the different quantiles of the ground truth

        n = gt_preds.shape[0]

        # Samples per bin
        samples_per_bin = n // num_bins

        # Get the indices for the bins
        indices = np.argsort(gt_preds)
        bin_indices = np.array_split(indices, num_bins)

        # Get the risk for the different bins
        bin_risks = []
        for bin_idx in bin_indices:
            bin_risks.append(self.get_risk(intervals[bin_idx], gt_preds[bin_idx]))

        return bin_risks

    # Compute the lambda value
    def compute_lambda(self, recon_preds, gt_preds):
        def lambda_func(lam):
            # Get the number of calibration points
            n = recon_preds.shape[0]

            # Get the interval
            intervals, _ = self.get_interval(recon_preds, lam)

            # Get the loss
            loss = self.get_risk(intervals, gt_preds)

            # Turn into a function that crosses zero
            loss = loss - ((n+1)/n*self.alpha - self.B/n)

            return loss

        # Get the lambda value
        self.lambda_hat = brentq(lambda_func, -25, 50)



    def conformal_inference(self, recon_preds):
        """
        Get the conformal interval and the lambda value
        :param recon_preds: (n, m) array of metrics for the reconstructions
        :return: (n, 2) array of conformal intervals
        """

        if self.lambda_hat is None:
            raise ValueError('Lambda hat value not computed yet')

        # Get the interval
        intervals, interval_size = self.get_interval(recon_preds, self.lambda_hat)

        return intervals, interval_size

    def get_pinball_loss(self, worst_bounds, gt_preds):
        """
        Get the pinball loss for the worst bounds
        :param worst_bounds: (n) array of the worst bounds
        :param gt_preds: (n) array of the ground truth metrics
        :return:
        """

        # Get the pinball loss
        if self.metric == 'psnr' or self.metric == 'ssim':
            pinball_loss = np.mean(np.maximum(gt_preds - worst_bounds, 0) * self.alpha + np.maximum(worst_bounds - gt_preds, 0) * (1 - self.alpha))
        else:
            pinball_loss = np.mean(np.maximum(gt_preds - worst_bounds, 0) * (1 - self.alpha) + np.maximum(worst_bounds - gt_preds, 0) * self.alpha)

        return pinball_loss


    # Single Monte Carlo trial
    def eval_single_trial(self, cal_recon_preds, cal_gt_preds, test_recon_preds, test_gt_preds, correlation=False, num_bins=5):
        """
        Get the conformal interval and the lambda value
        :param cal_recon_preds: (n, m) array of metrics for the reconstructions
        :param cal_gt_preds: (n, p) array of ground truth metrics
        :param test_recon_preds: (b, m) array of metrics for the reconstructions
        :param test_gt_preds: (b, p) array of ground truth metrics
        :return:
        """

        # Compute the lambda value
        self.compute_lambda(cal_recon_preds, cal_gt_preds)

        # Get the conformal interval
        intervals, interval_size = self.conformal_inference(test_recon_preds)

        # Get the risk on the test set
        risk = self.get_risk(intervals, test_gt_preds)

        # Get the conditional risk for different bins
        bin_risks = self.get_bin_risk(intervals, test_gt_preds, num_bins=num_bins)

        # Get the bound on the interval
        if self.metric == 'psnr' or self.metric == 'ssim':
            worst_bound = intervals[:,0]
        else:
            worst_bound = intervals[:,1]
        #worst_bound = intervals[:, 1]
        pinball_loss = self.get_pinball_loss(worst_bound, test_gt_preds)

        if correlation:
            corr = np.corrcoef(test_gt_preds, worst_bound)[0, 1]
            return risk, interval_size, worst_bound, corr, bin_risks, pinball_loss

        return risk, interval_size, worst_bound, bin_risks, pinball_loss


    # Perform many Monte Carlo trials
    def eval_many_trials(self, recon_preds, gt_preds, num_trials, calib_split=0.7, num_c=None, correlation=False, num_bins=5):


        # Get the number of samples
        n = recon_preds.shape[0]

        # Get the number of calibration samples
        n_cal = int(n * calib_split)

        # Collect all the risks for all the trials
        risks = []
        mean_interval_sizes = []
        mean_worst_bounds = []
        corrs = []
        bin_risks_all = []
        pinball_losses = []

        np.random.seed(0) # Ensures each method uses the same random trials

        for i in tqdm(range(num_trials)):

            # Get the indices for the calibration and test sets
            indices = np.random.permutation(n)
            cal_indices = indices[:n_cal]
            test_indices = indices[n_cal:]

            # if i ==0:
            #     print('Calibration Indices:', cal_indices[0:5])

            # Get the indices for the posterior samples to use
            if num_c is not None:
                post_indices = np.random.permutation(recon_preds.shape[1])[:num_c]

                # Get the calibration and test sets
                cal_recon_preds = recon_preds[cal_indices][:, post_indices]
                cal_gt_preds = gt_preds[cal_indices]
                test_recon_preds = recon_preds[test_indices][:, post_indices]
                test_gt_preds = gt_preds[test_indices]
                #print(cal_recon_preds.shape)

            # Get the calibration and test sets
            else:
                cal_recon_preds = recon_preds[cal_indices]
                cal_gt_preds = gt_preds[cal_indices]
                test_recon_preds = recon_preds[test_indices]
                test_gt_preds = gt_preds[test_indices]

            # Get the risk on the test set
            risk, interval_sizes, worst_bound, corr, bin_risks, pinball_loss = self.eval_single_trial(cal_recon_preds, cal_gt_preds, test_recon_preds, test_gt_preds, correlation=True, num_bins=num_bins)

            risks.append(risk)
            mean_interval_sizes.append(interval_sizes.mean())
            mean_worst_bounds.append(worst_bound.mean())
            corrs.append(corr)
            bin_risks_all.append(bin_risks)
            pinball_losses.append(pinball_loss)

        if correlation:
            return risks, mean_interval_sizes, mean_worst_bounds, corrs, bin_risks_all, pinball_losses
        return risks, mean_interval_sizes, mean_worst_bounds, bin_risks_all, pinball_losses


    def fit_preprocessor(self, train_recon_preds, train_gt_preds, **kwargs):
        """
        Fit the feature preprocessor
        :param train_recon_preds: (n, m) array of metrics for the reconstructions
        :param train_gt_preds: (n,1) array of ground truth metrics
        :return: scale and shift
        """

        self.feature_preprocessor.fit(train_recon_preds, train_gt_preds, **kwargs)


class PSNR:
    def __init__(self, device='cuda'):
        self.device = device

    def __call__(self, recon, gt):
        '''
        Compute the PSNR between the reconstruction and the ground truth
        :param recon: (n, c, h, w) tensor of the reconstruction
        :param gt: (n, c, h, w) tensor of the ground truth
        :return:
        '''

        if isinstance(recon, np.ndarray):
            recon = torch.tensor(recon)
        if isinstance(gt, np.ndarray):
            gt = torch.tensor(gt)

        recon = recon.to(self.device)
        gt = gt.to(self.device)

        se = (recon - gt) ** 2
        mse = torch.mean(se, dim=(1, 2, 3))
        max_intensity = torch.max(gt.reshape(gt.shape[0], -1), dim=1)[0]
        psnr = 10 * torch.log10(max_intensity**2 / mse)

        return psnr


def normalize( x):
    # Normalize to be between 0 and 1
    flattened_imgs = x.view(x.shape[0], -1)
    min_val, _ = torch.min(flattened_imgs, dim=1)
    max_val, _ = torch.max(flattened_imgs, dim=1)
    x = (x - min_val.view(-1, 1, 1, 1)) / (max_val.view(-1, 1, 1, 1) - min_val.view(-1, 1, 1, 1))

    return x

def to_rgb(x):
    '''
    Repeat the channel dimensions to make the tensor RGB
    :param x: (b, 1, h, w) tensor
    :return:
    '''

    # Repeat the channel dimensions
    x = x.repeat(1, 3, 1, 1)
    return x


def single_indicator_risk(intervals, gt_preds):
    """
    See if the gt_preds for each sample is within the interval
    :param intervals: (n, 2) array of intervals
    :param gt_preds: (n) array of ground truth metrics
    :return:
    """

    # Get a ground truth prediction for each sample
    # If there are multiple dimensions, they're all the same so take the first one (this mistakes corrected)
    if len(gt_preds.shape) > 1:
        z_gt = gt_preds[:, 0]
    else:
        z_gt = gt_preds

    # Find the number of z_gt that are outside the interval
    max_out_interval = np.logical_or(intervals[:, 0] > z_gt, z_gt > intervals[:, 1])

    # Get the number of samples in the interval
    loss = sum(max_out_interval) / len(z_gt)

    return loss



# Class that preprocesses the quality samples before conformalizing
class FeaturePreprocessor:

    def __init__(self, alpha, method='quantile', all_features=False, device='cuda', k=5):
        self.alpha = alpha
        self.method = method
        self.all_features = all_features
        self.device = device
        self.k = k


    def fit(self, x, y, **kwargs):
        """
        Fit the feature preprocessor
        :param x: (n, d) array of input data
        :param y: (n) array of output data
        :return:
        """

        if 'hyperparams' in kwargs:
            omegas = kwargs['hyperparams']['omegas']
            gammas = kwargs['hyperparams']['gammas']
            rhos = kwargs['hyperparams']['rhos']

        else:
            omegas = [0.0001, 0.001, 0.01, 0.1,  10]
            gammas = [0.01, 1.0, 10.0, 100.0]
            rhos = [10**-i for i in range(4,8)]

        # No fitting required
        if self.method== 'quantile' or self.method == 'mean' or self.method == 'nonadaptive':
            return

        # Fit a regressor
        elif self.method == 'lad' or self.method == 'qr' or self.method == 'linear':
            if self.method == 'lad':
                self.regressor = QuantileRegressor(quantile=0.5, alpha=0.0, solver="highs")
            elif self.method == 'qr':
                self.regressor = QuantileRegressor(quantile=self.alpha, alpha=0.0, solver="highs")
            elif self.method == 'linear':
                self.regressor = LinearRegression()

            # Get the quantile of the features
            if not self.all_features:
                q = np.quantile(x, self.alpha, axis=1)
            else:
                p = x.shape[1]
                q = x.flatten()
                y = y.repeat(p)

            self.regressor.fit(q.reshape(-1,1), y)


        elif self.method == 'regression':
            self.regressor = KernelQuantileRegressor(alpha=self.alpha, device=self.device, kernel = 'spline', quantile=False)
            omega, gamma, rho = self.regressor.kfold_cv_random(x, y, omegas, gammas, rhos, k=self.k)
            self.regressor.fit(x, y, omega=omega, gamma=gamma, rho=rho)

        elif self.method == 'xgboost':
            x = np.sort(x, axis=1)
            self.regressor = XGBRegressor(objective='reg:quantileerror', quantile_alpha=self.alpha, random_state=0)
            self.regressor.fit(x, y)


        else:
            raise ValueError('Method not implemented')



    def predict(self, x):

        if self.method == 'quantile':
            x = np.quantile(x, self.alpha, axis=1)
        elif self.method == 'mean':
            x = np.mean(x, axis=1)
        elif self.method == 'lad' or self.method == 'qr' or self.method == 'linear':
            if not self.all_features:
                q = np.quantile(x, self.alpha, axis=1)
                x = self.regressor.predict(q.reshape(-1,1))
            else:
                q = x.flatten()
                x = self.regressor.predict(q.reshape(-1,1))
                x = np.quantile(x, self.alpha, axis=1)
        elif self.method == 'nonadaptive':
            x = np.zeros(x.shape[0])

        #elif self.method == 'kernel' or self.method == 'quantile_kernel' or self.method == 'banded_kernel':
        else:
            if self.method == 'xgboost':
                x = np.sort(x, axis=1)
            x = self.regressor.predict(x)

        return x


# Class for the regression model with kernels. Performs the quadratic program
class KernelQuantileRegressor:
    def __init__(self, alpha=0.05, device = 'cuda', quantile=False, kernel='rbf'):
        self.alpha = alpha
        self.theta = None
        self.b = None
        self.omega = None
        self.gamma = None
        self.rho = None
        self.x_train = None
        self.device = device
        self.quantile = quantile
        self.kernel = kernel

    def fit(self, x, y, omega, gamma, rho, K=None):
        """
        Fit the quantile regressor
        alpha: quantile level
        omega: kernel bandwidth
        rho: weight for quantile regression loss
        gamma: weight for bound size
        :param x: (n, d) array of input data
        :param y: (n) array of output data
        :return:
        """

        # Get the number of samples
        n = x.shape[0]
        C = 1/(rho*n) # Regularization on quantile regression
        alpha = self.alpha

        # Make gamma negative if it's a lower bound like for PSNR or SSIM
        if alpha < 0.5:
            gamma = -gamma # Encourages larger predictions for lower quantiles

        x = x.astype(np.double)
        y = y.astype(np.double)

        # Sort the reconstruction metrics for each sample in ascending order
        if self.quantile:
            x = np.quantile(x, alpha, axis=1).reshape(-1,1)
        else:
            x = np.sort(x, axis=1)

        # Compute the RBF kernel
        if K is None:
            K = self.compute_kernel_matrix(x, x, omega).cpu().numpy()

        # K = K + np.eye(n)*1e-6 # Add a small amount to the diagonal to make it positive definite

        # Prep the other quadratic programming terms
        zeros_mat = np.zeros((n, n))
        In = np.eye(n)
        ones_vec = np.ones(n)
        zeros_vec = np.zeros(n)

        P = K
        q = torch.matmul(torch.from_numpy(K).to('cuda'),
                         C * gamma * torch.ones(n, dtype=torch.double).to('cuda')).cpu().numpy() - y
        G = np.block([[In], [-In]])  # Inequality constraint matrix
        h = np.concatenate([C * alpha * ones_vec, -C * (alpha - 1) * ones_vec])  # Inequality constraint vector
        A = np.ones((1, n))
        b = np.array([-C*gamma*n])


        # Solve the quadratic program
        try:
            theta = solve_qp(P, q, G=G, h=h, A=A, b=b, solver='cvxopt', verbose=False)  # , lb=-np.inf*np.ones(2*n), ub=np.inf*np.ones(2*n))

        except:
            #print(f'Problem with optimization for omega: {omega}, alpha: {alpha}, gamma: {gamma}, rho: {rho}')
            return None, None

        if theta is None:
            #print(f'Optimization for omega: {omega}, alpha: {alpha}, gamma: {gamma}, rho: {rho} did not work')
            return None, None
        else:

            idx = np.argwhere((np.abs(theta - alpha * C) > 1e-5) & (np.abs(
                theta - (alpha - 1) * C) > 1e-5))  # Find where theta1+theta2 != alpha*C or (alpha-1)*C

            if len(idx) == 0:
                idx  = np.argmax( np.abs(theta - alpha * C) * np.abs(theta - (alpha - 1) * C) ) # Find index with largest differences


            intercept = torch.from_numpy(y[idx.flatten()]).to('cuda') - torch.matmul(
                torch.from_numpy(K[idx.flatten(), :]).to('cuda'),
                torch.from_numpy(theta + C * gamma * ones_vec).to('cuda'))
            if intercept.shape[0] > 1:
                if torch.abs(intercept[0] - intercept[1]) > 1e-5:
                    intercept = torch.mean(intercept)
                else:
                    intercept = intercept[0]  # Note: all b should be the same

            self.theta = torch.from_numpy(theta)
            self.b = intercept
            self.omega = omega
            self.gamma = gamma
            self.rho = rho
            self.x_train = x

            return theta, intercept


    def predict(self, x, theta=None, b=None, omega = None, K = None):
        # Use the input theta and b if given
        if theta is None:
            if self.theta is None:
                raise ValueError('Fit model or input theta and b')
            else:
                theta = self.theta
                b = self.b
                omega = self.omega

        if isinstance(theta, np.ndarray):
            theta = torch.from_numpy(theta)
        theta = theta.to(self.device)
        if isinstance(b, np.ndarray):
            b = torch.from_numpy(b).to(self.device)
        b = b.to(self.device)

        # Get the kernel
        x = x.astype(np.double)
        if self.quantile:
            x = np.quantile(x, self.alpha, axis=1).reshape(-1,1)
            x_train = np.quantile(self.x_train, self.alpha, axis=1).reshape(-1,1)
        else:
            x = np.sort(x, axis=1)
            x_train = np.sort(self.x_train, axis=1)
        if K is None:
            K = self.compute_kernel_matrix(x, x_train, omega)
        if isinstance(K, np.ndarray):
            K = torch.from_numpy(K)

        K = K.to(self.device)

        # Get the function output
        #out =  torch.matmul(K, theta[:x_train.shape[0]]) + b #np.dot(K, theta[:x_train.shape[0]]) + b
        C = 1 / (self.rho * x_train.shape[0])
        out = torch.matmul(K, theta + C*self.gamma) + b

        return out.cpu().numpy()



    def compute_kernel_matrix(self, x, x_prime, omega):
        """
        Compute the RBF kernel
        :param x: (n, d) array of input data
        :param x_prime: (m, d) array of input data
        :param omega: kernel bandwidth
        :return: (n, m) array of kernel values

        Note: Don't forget to sort the inputs before putting into this function
        """


        d1 = x.shape[1]
        d2 = x_prime.shape[1]
        #m1 = x.mean(axis=0)  # .reshape(-1,1)
        m2 = x_prime.mean(axis=0)  # .reshape(-1,1)
        x = (x - m2) / np.sqrt(d2)
        x_prime = (x_prime - m2) / np.sqrt(d2)

        if self.kernel == 'rbf':
            # Much faster way to compute the kernel
            diff = torch.from_numpy(x[:, np.newaxis]).to(self.device) - torch.from_numpy(x_prime).to(self.device)
            dists = torch.norm(diff, dim=-1) #np.linalg.norm(diff, axis=-1)
            K = torch.exp(-omega * dists ** 2) #np.exp(-omega * dists ** 2)

        elif self.kernel == 'poly':
            dot = torch.matmul(torch.from_numpy(x).to(self.device), torch.from_numpy(x_prime).to(self.device).T)
            K = dot + dot**2 + dot**3 + dot**4

        elif self.kernel == 'spline':
            num_knots = 2

            # Get the locations of the knots
            t2 = np.quantile(np.mean(x_prime, axis=1), np.arange(0, 1, 1 / (num_knots + 1))[1:])
            phi1_new_feats1 = np.maximum(0, x - t2[0])
            phi1_new_feats2 = np.maximum(0, x - t2[1])
            phi1 = np.concatenate([x, phi1_new_feats1, phi1_new_feats2], axis=1)

            phi2_new_feats1 = np.maximum(0, x_prime - t2[0])
            phi2_new_feats2 = np.maximum(0, x_prime - t2[1])
            phi2 = np.concatenate([x_prime, phi2_new_feats1, phi2_new_feats2], axis=1)

            K = torch.matmul(torch.from_numpy(phi1).to(self.device), torch.from_numpy(phi2).to(self.device).T)

        return K


    def kfold_cv_random(self, x, y, omegas, gammas, rhos, k=10):
        """
        Perform k-fold cross validation to find the best hyperparameters
        :param x: (n, d) array of input data
        :param y: (n) array of output data
        :param omegas: list of kernel bandwidths to try
        :param gammas: list of weights for bound size
        :param rhos: list of weights for quantile regression
        :param k: number of folds
        :return: best hyperparameters
        """
        print('Performing k-fold cross validation for kernel quantile regression...')

        # Get the number of samples
        n = x.shape[0]

        # Samples per fold
        n_fold = n // k

        # Sort based on the inputs

        rng = np.random.default_rng()
        indices = rng.permutation(n)
        fold_indices = np.array_split(indices, k)

        # Track the losses across folds and hyperparameters
        losses = np.zeros((k, len(omegas), len(gammas), len(rhos)))

        for i in tqdm(range(k)):

            # Get the training and validation data
            train_indices = np.concatenate([fold_indices[j] for j in range(k) if j != i])
            val_indices = fold_indices[i]

            x = x.astype(np.double)
            y = y.astype(np.double)

            x_train = x[train_indices]
            y_train = y[train_indices]
            x_val = x[val_indices]
            y_val = y[val_indices]

            if self.quantile:
                x_train = np.quantile(x_train, self.alpha, axis=1).reshape(-1,1)
                x_val = np.quantile(x_val, self.alpha, axis=1).reshape(-1,1)
            else:
                x_train = np.sort(x_train, axis=1)
                x_val = np.sort(x_val, axis=1)

            for o, omega in enumerate(omegas):
                for g, gamma in enumerate(gammas):

                    #Compute the kernels once for all rhos
                    K = self.compute_kernel_matrix(x_train, x_train, omega).cpu().numpy()
                    K_test = self.compute_kernel_matrix(x_val, x_train, omega).cpu().numpy()

                    for r, rho in enumerate(rhos):
                        loss = 0
                        #loss = []

                        # Fit the model
                        theta, b = self.fit(x_train, y_train, omega, gamma, rho, K)

                        if theta is None:
                            losses[i, o, g, r] = np.inf
                            break

                        # Get the loss
                        y_pred = self.predict(x_val, theta, b, omega, K_test)

                        losses[i, o, g, r] = np.mean(self.alpha * np.maximum(y_val - y_pred,0) + (1 - self.alpha) * np.maximum(y_pred - y_val, 0))

        losses = np.array(losses)
        # Average across folds
        losses = np.mean(losses, axis=0)

        # Find the indices of the best hyperparameters
        idx = np.unravel_index(np.argmin(losses), losses.shape)
        best_hyperparams = (omegas[idx[0]], gammas[idx[1]], rhos[idx[2]])

        print(f'Best hyperparameters: {best_hyperparams}')
        print(f'Best loss: {losses[idx]:.3f}')

        self.omega = best_hyperparams[0]
        self.gamma = best_hyperparams[1]
        self.rho = best_hyperparams[2]


        return best_hyperparams



















