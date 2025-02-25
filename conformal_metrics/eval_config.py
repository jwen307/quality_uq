import numpy as np

class Config():

    def __init__(self):
        self.config = {
            'alpha': 0.05, # Error rate
            'num_trials': 100, # Number of Monte Carlo trials #TODO: Change back
            'calib_split': 0.7, # Proportion of data to use for calibration
            'load_ckpt_dir': '/storage/logs/MulticoilCNF/version_0/', # Directory to load model from
            'feature_preprocess_method': 'quantile', # 'nonadaptive', 'quantile', 'regression'
            'kfold_hyperparams': {'omegas': [0],
               'gammas': [0.0],
               'rhos': np.logspace(-8,1,40)}
        }