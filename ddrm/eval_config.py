import numpy as np

class Config():

    def __init__(self):
        self.config = {
            'alpha': 0.05, # Error rate
            'num_trials': 100, # Number of Monte Carlo trials #TODO: Change back
            'calib_split': 0.7, # Proportion of data to use for calibration
            'num_train': 1000, # Number of training samples
            'images_dir': '/storage/logs/ddrm/image_samples/conformal/0.75/', # Directory where posteriors are stored
            'feature_preprocess_method': 'regression', # 'nonadaptive', 'quantile', 'regression'
            'kfold_hyperparams': {'omegas': [0],
               'gammas': [0.0],
               'rhos': np.logspace(-8,1,40)}
        }