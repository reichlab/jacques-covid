import math
import numpy as np
import tensorflow as tf

from jacques import kcqe

from .featurize import featurize_data

class kcqe_rollmean():
    def __init__(self,
                 x_kernel = 'gaussian_diag',
                 lags = [7, 14, 21],
                 h = 1,
                 tau = tf.constant(np.array([0.1, 0.5, 0.9]))) -> None:
        self.x_kernel = x_kernel
        self.lags = lags
        self.h = h
        self.tau = tau

    
    def fit(self,
            data,
            init_param_vec = None,
            num_epochs = 10,
            verbose = False,
            tau = None):
        if tau is None:
            tau = self.tau
        data['fourth_rt_rate'] = data['rate'] ** 0.25
        self.x_train_val, self.y_train_val, self.x_T = featurize_data(
            data,
            target_var='rate',
            h=self.h,
            features = [{
                    'fun': 'moving_average',
                    'args': {'target_var': 'fourth_rt_rate', 'num_days': 7}
                },
                {
                    'fun': 'lagged_values',
                    'args': {'target_var': 'moving_avg_7_fourth_rt_rate', 'lags': self.lags}
                }
            ])
        self.kcqe_obj = kcqe.KCQE(x_kernel = self.x_kernel, p=len(self.lags) + 1)
        block_size = 21
        num_blocks = math.floor(self.y_train_val.shape[1]/block_size)
        self.generator = self.kcqe_obj.generator(self.x_train_val,
                                                 self.y_train_val,
                                                 batch_size = num_blocks,
                                                 block_size = block_size)
        
        if init_param_vec is None:
            init_param_vec = tf.constant(np.zeros(self.kcqe_obj.n_param), dtype=np.float32)
        
        self.param_vec = self.kcqe_obj.fit(
            xval_batch_gen = self.generator,
            num_blocks = num_blocks,
            tau=tau,
            optim_method='adam',
            num_epochs=num_epochs,
            learning_rate=0.1,
            init_param_vec=init_param_vec,
            verbose = verbose)


    def predict(self, tau = None):
        if tau is None:
            tau = self.tau
        q_hat = self.kcqe_obj.predict(self.param_vec,
                                      x_train=self.x_train_val,
                                      y_train=self.y_train_val,
                                      x_test=self.x_T,
                                      tau=tau)
        return q_hat

