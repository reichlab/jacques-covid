import math
import numpy as np
import tensorflow as tf

from jacques import kcqe

from .featurize import featurize_data

class kcqe_rollmean():
  def __call__(self,
               data,
               x_kernel = 'gaussian_diag',
               lags = [7, 14],
               tau = tf.constant(np.array([0.1, 0.5, 0.9])),
               num_epochs = 10,
               verbose = False):
    data['fourth_rt_rate'] = data['rate'] ** 0.25
    x_train_val, y_train_val, x_T = featurize_data(data, target_var="rate", h=1,
      features = [{
          "fun": 'moving_average',
          "args": {'target_var': "fourth_rt_rate", 'num_days': 7}
        },
        {
          "fun": 'lagged_values',
          "args": {'target_var': "moving_avg_7_fourth_rt_rate", 'lags': lags}
        }
      ])
    kcqe_obj = kcqe.KCQE(p=len(lags) + 1)
    block_size = 21
    generator = kcqe_obj.generator(x_train_val, y_train_val,
                                   batch_size = 1,
                                   block_size = block_size)
    self.param_vec = kcqe_obj.fit(xval_batch_gen = generator,
      num_blocks = math.floor(y_train_val.shape[1]/block_size),
      tau=tau,
      optim_method="adam",
      num_epochs=num_epochs,
      learning_rate=0.1,
      init_param_vec=tf.constant(np.zeros(len(lags) + 2)),
      verbose = True)
    
    q_hat = kcqe_obj.predict(self.param_vec,
                             x_train=x_train_val,
                             y_train=y_train_val,
                             x_test=x_T,
                             tau=tau)
    
    return q_hat

