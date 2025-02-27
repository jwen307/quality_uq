import numpy as np

class Config():

    def __init__(self):
        self.config = {
            'alpha': 0.05, # Error rate
            'num_trials': 1000, # Number of Monte Carlo trials #TODO: Change back
            'calib_split': 0.7, # Proportion of data to use for calibration
            'load_recovery_ckpt': '/storage/logs/VarNet/version_0/', # Directory to load recovery model from
            'load_posterior_ckpt': '/storage/logs/MulticoilCNF/version_0/', # Directory to load posterior model from

            'feature_preprocess_method': 'nonadaptive', # 'nonadaptive', 'quantile', 'regression'
            'kfold_hyperparams': {'omegas': [0],
               'gammas': [0.0],
               'rhos': np.logspace(-8,1,40)}
        }