import numpy as np
import pandas as pd
import math


def trailing_taylor_coefs_taylor_update(loc_data,
                                        target_var,
                                        taylor_degree = 1,
                                        taylor_innovations = 1,
                                        update_frequency = 1,
                                        window = 21,
                                        include_dow = True,
                                        ew_span = 14,
                                        realtime = True):
    '''
    Estimate the parameters of a Taylor polynomial fit to a rolling
    trailing window, with the coefficients in consecutive windows
    updated according to a Taylor process with noise.
    
    Parameters
    ----------
    loc_data: a pandas data frame with data for one location
    target_var: variable for which to calculate neighborhoods
    taylor_degree: degree of the Taylor polynomial
    taylor_innovations: 'all' or 'highest'
    window: window size
    ew_span: span for exponential weighting of observations.
        No weighting is done if ew_span is None.
    realtime: if True, get estimates at each time point as though in real time
    
    Returns
    -------
    Data frame with the same number of rows as the input data and columns
    `target_var + '_taylor_' + d` for each degree d in 0, 1, 2
    
    Notes
    -----
    In window t, the updated coefficients are A @ beta_{t-1}, where A is
    [ 1  1  0.5  ...  1 / D!
      0  1  1    ...  1 / (D-1)!
      ...
      0  0  0    ...  1           ]
      
    With D = 2, this is
    
    [ 1  1  0.5 ]                [ 0 ]
    [ 0  1  1   ] @ beta_{t-1} + [ 0 ]
    [ 0  0  1   ]                [ gamma_t ]
    
    Suppose we start with a vector alpha of length taylor_degree + [T - (taylor_degree - 1)] + 6
    We can convert this to the vector of taylor coefficients by setting
    beta_1 = [I_3  0  ...  0 ] @ alpha
    
             [ 1  1  0.5 ]                [ 0  0  0  0  0 ... 0 ]
    beta_2 = [ 0  1  1   ] @ beta_{t-1} + [ 0  0  0  0  0 ... 0 ] @ alpha
             [ 0  0  1   ]                [ 0  0  0  1  0 ... 0 ]
    
             [ 1  1  0.5  0 ... 0]
           = [ 0  1  1    0 ... 0] @ alpha
             [ 0  0  1    1 ... 0]
    
             [ 1  1  0.5 ]                [ 0  0  0  0  0 ... 0 ]
    beta_3 = [ 0  1  1   ] @ beta_{t-1} + [ 0  0  0  0  0 ... 0 ] @ alpha
             [ 0  0  1   ]                [ 0  0  0  0  1 ... 0 ]
    
             [ 1  1  0.5  0  0 ... 0]
           = [ 0  1  1    0  0 ... 0] @ alpha
             [ 0  0  1    0  1 ... 0]
    
    The actual X used is the X of Taylor basis functions times the above matrix premultiplying alpha.
    '''
    loc_data = loc_data[[target_var] + ['date']]
    for l in range(window):
        loc_data[target_var + '_lag_' + str(l)] = loc_data[[target_var]].shift(l)
        loc_data['dow_lag_' + str(l)] = loc_data['date'].dt.dayofweek.shift(l)

    loc_data_nona = loc_data.dropna()
    
    if update_frequency == 7:
        loc_data_nona = loc_data_nona[loc_data_nona.dow_lag_0 == 0]

    lagged_vars = [target_var + '_lag_' + str(l) for l in range(window)]
    y = loc_data_nona[lagged_vars].values.astype('float64')
    y = y.reshape(math.prod(y.shape))
    
    lagged_dow_vars = ['dow_lag_' + str(l) for l in range(window)]
    dow = loc_data_nona[lagged_dow_vars].values
    dow = dow.reshape(math.prod(dow.shape))
    
    # basis functions for separate Taylor polynomials per window
    if taylor_degree == 1:
        one_taylor_X = np.concatenate(
            [
                np.ones((window, 1)),
                np.expand_dims(-np.arange(window), -1)
            ],
            axis = 1
        )
        taylor_shift = np.array([[1., 1.], [0., 1.]])
    elif taylor_degree == 2:
        one_taylor_X = np.concatenate(
            [
                np.ones((window, 1)),
                np.expand_dims(-np.arange(window), -1),
                np.expand_dims(+0.5 * np.arange(window)**2, -1)
            ],
            axis = 1
        )
        taylor_shift = np.array([[1., 1., 0.5], [0., 1., 1.], [0., 0., 1.]])
    elif taylor_degree == 3:
        one_taylor_X = np.concatenate(
            [
                np.ones((window, 1)),
                np.expand_dims(-np.arange(window), -1),
                np.expand_dims(+0.5 * np.arange(window)**2, -1),
                np.expand_dims(-(1. / 6.) * np.arange(window)**3, -1)
            ],
            axis = 1
        )
        taylor_shift = np.array([[1., 1., 0.5, 1/6.], [0., 1., 1., 0.5], [0., 0., 1., 1.], [0., 0., 0., 1.]])
    
    n_orig = loc_data_nona.shape[0]
    n_taylor_coef = taylor_degree + 1
    taylor_X = [
        np.concatenate(
            [
                np.zeros((window, i * n_taylor_coef)),
                one_taylor_X,
                np.zeros((window, (n_orig - i - 1) * n_taylor_coef))
            ],
            axis = 1
        ) \
            for i in range(n_orig)
    ]
    taylor_X = np.concatenate(taylor_X, axis = 0)
    # print("taylor_X")
    # print(taylor_X)
    
    # if taylor_innovations == 'highest':
    obs_i_shift = np.concatenate(
        [np.eye(n_taylor_coef), np.zeros((n_taylor_coef, (n_orig - 1) * taylor_innovations))],
        axis = 1
    )
    taylor_coef_shift = [ obs_i_shift ]
    
    for i in range(1, n_orig):
        obs_i_shift = np.matmul(taylor_shift, obs_i_shift)
        for j in range(taylor_innovations):
            obs_i_shift[n_taylor_coef - j - 1, n_taylor_coef + i * taylor_innovations - j - 1] += 1.
            # print("obs_i_shift, j = " + str(j))
            # print(obs_i_shift)
        taylor_coef_shift = taylor_coef_shift + [
            obs_i_shift
        ]
    
    taylor_coef_shift = np.concatenate(taylor_coef_shift, axis = 0)
    # elif taylor_innovations == 'all':
    #     taylor_coef_shift = np.identity(taylor_X.shape[1])

    # basis functions for day of week effects
    # we require the effects to sum to 0: \sum_{i=0}^6 gamma_i = 0.
    # then gamma_6 = - \sum_{i=0}^5 gamma_i
    # dow_X has 6 columns with indicators for day of weeks 0 through 5,
    # all values -1 in rows where day of week is 6
    dow_X = np.zeros((len(dow), 6))
    for i in range(6):
        dow_X[np.where(dow == i), i] = 1.
    
    dow_X[np.where(dow == 6), :] = -1.
    
    # combine taylor_X and dow_X
    if realtime:
        beta_hat_taylor_rows = []
        for i in range(n_orig):
            # print("taylor_X_i input")
            # print(taylor_X[:((i+1)*window), :])
            # print(taylor_coef_shift[:, :(n_taylor_coef + i * taylor_innovations)])
            taylor_X_i = np.matmul(taylor_X, taylor_coef_shift)[:((i+1)*window), :(n_taylor_coef + i * taylor_innovations)]
            # if taylor_innovations == 'highest':
            #     taylor_X_i = np.matmul(taylor_X, taylor_coef_shift)[:((i+1)*window), :(n_taylor_coef + i)]
            # else:
            #     taylor_X_i = np.matmul(taylor_X, taylor_coef_shift)[:((i+1)*window), :(n_taylor_coef * (i + 1))]
            
            if include_dow:
                X_i = np.concatenate(
                    [
                        taylor_X_i,
                        dow_X[:((i+1)*window), :]
                    ],
                    axis=1
                )
            else:
                X_i = taylor_X_i
            
            # print("unweighted X")
            # print(X_i)
            
            y_i = y[:((i+1)*window)]

            if ew_span is not None:
                # organize exponential weighted observation weights
                ew_alpha = 2 / (ew_span + 1)
                orig_obs_weights = ew_alpha * (1 - ew_alpha)**np.arange(i + window - 1, -1, -1)
                windowed_obs_weights = np.concatenate(
                    [orig_obs_weights[j:(window + j)][::-1] for j in range(i + 1)],
                    axis = 0
                )
                W = np.diag(np.sqrt(windowed_obs_weights))
                
                # update X_i and y_i to incorporate weights
                X_i = np.matmul(W, X_i)
                y_i = np.matmul(W, y_i)

            # print("X_i")
            # print(X_i)
            # print("y_i")
            # print(y_i)
            beta_hat = np.linalg.lstsq(X_i, y_i, rcond=None)[0]
            # print("beta_hat")
            # print(beta_hat)
            
            if include_dow:
                beta_hat_taylor = beta_hat[:-6]
            else:
                beta_hat_taylor = beta_hat
            
            beta_hat_taylor = np.matmul(
                taylor_coef_shift[:(n_taylor_coef * (i + 1)), :(n_taylor_coef + i * taylor_innovations)],
                beta_hat_taylor)
            # if taylor_innovations == 'highest':
                # beta_hat_taylor = np.matmul(
                #     taylor_coef_shift[:(n_taylor_coef * (i + 1)), :(n_taylor_coef + i)],
                #     beta_hat[:-6])
            # else:
            #     beta_hat_taylor = beta_hat[:-6]
            
            beta_hat_taylor = beta_hat_taylor.reshape(
                (beta_hat_taylor.shape[0] // n_taylor_coef, n_taylor_coef)
            )
            # print("beta_hat_taylor")
            # print(beta_hat_taylor)
            
            beta_hat_taylor_rows = beta_hat_taylor_rows + \
                [ beta_hat_taylor[i:(i + 1), :] ]
        
        beta_hat_taylor = np.concatenate(beta_hat_taylor_rows, axis = 0)
        # print("final beta_hat_taylor")
        # print(beta_hat_taylor)
    else:
        taylor_X = np.matmul(taylor_X, taylor_coef_shift)
        
        X = np.concatenate([taylor_X, dow_X], axis=1)
        
        beta_hat = np.linalg.lstsq(X, y, rcond=None)[0]
        
        beta_hat_taylor = np.matmul(taylor_coef_shift, beta_hat[:-6])
        beta_hat_taylor = beta_hat_taylor.reshape(
            (beta_hat_taylor.shape[0] // n_taylor_coef, n_taylor_coef)
        )
    
    # if include_dow:
    #     beta_hat_dow = beta_hat[-6:]
    #     beta_hat_dow = np.concatenate(
    #         [beta_hat_dow, -np.sum(beta_hat_dow, keepdims=True)],
    #         axis = 0
    #     )
    # else:
    #     beta_hat_dow = []
    
    # return beta_hat, beta_hat_taylor, beta_hat_dow
    return pd.DataFrame({
        target_var + '_taylor_' + str(d): np.concatenate([np.full(window - 1, np.nan), beta_hat_taylor[:, d]], axis = 0) \
            for d in range(n_taylor_coef)
    })



def gp_smooth(t, y,
              amp_init = None,
              ls_init = None,
              theta_init = None,
              sigma_init = None,
              n_iter=20000):
  """
  Obtain a smoothed estimate of the probability of success in a Binomial
  observation model by fitting a latent Gaussian process.

  Parameters
  ----------
  t: time values at which we have observations
  y: observed values

  Returns
  -------
  Numpy array of same length as t with smoothed values at each time point
  """
  if amp_init is None:
    amp_init = np.float64(1.0)
  
  if ls_init is None:
    ls_init = np.float64(1.0)
  
  if theta_init is None:
    theta_init = y
  
  if sigma_init is None:
    sigma_init = np.float64(1.0)
  
  def build_gp(amp, ls):
    """
    Construct a GP model given values for amplitude and length scale
    """
    kernel = psd_kernels.ExponentiatedQuadratic(
      amplitude=amp,
      length_scale=ls
    )
    
    gp = tfd.GaussianProcess(kernel, np.expand_dims(t, -1))
    
    return gp
  
  # Set up tensorflow variables for amplitude, length scale, and logit of theta
  amp = tf.Variable(
    initial_value=amp_init,
    name='amp',
    dtype=np.float64
  )
  
  ls = tf.Variable(
    initial_value=ls_init,
    name='ls',
    dtype=np.float64
  )
  
  theta = tf.Variable(
    initial_value=theta_init,
    name='logit_theta',
    dtype=np.float64
  )
  
  sigma = tf.Variable(
    initial_value=sigma_init,
    name='sigma',
    dtype=np.float64
  )
  
  # Define full model
  model_dict = {
    'amp': tfd.HalfCauchy(
      loc=np.float64(0.0),
      scale=np.float64(100.0)),
    'ls': tfd.HalfCauchy(
      loc=np.float64(0.0),
      scale=np.float64(1.0)),
    'theta': build_gp,
    'sigma': tfd.HalfCauchy(
      loc=np.float64(0.0),
      scale=np.float64(1.0)),
    'y': lambda theta, sigma: tfd.Normal(loc=theta, scale=sigma)
  }
  
  joint_model = tfd.JointDistributionNamed(model_dict,
                                          batch_ndims=0,
                                          use_vectorized_map=True)
  
  # Obtain MAP estimates of model parameters
  trainable_variables = [amp, ls, theta, sigma]
  
  optimizer = tf.optimizers.Adam()
  
  @tf.function
  def optimize():
    with tf.GradientTape() as tape:
      loss = -tf.math.reduce_sum(joint_model.log_prob({'amp': amp, 'ls': ls, 'theta': theta, 'sigma': sigma, 'y': y}))
    grads = tape.gradient(loss, trainable_variables)
    optimizer.apply_gradients(zip(grads, trainable_variables))
    return loss
  
  for i in range(n_iter):
    neg_log_likelihood = optimize()
    if i % 1000 == 0:
      print("Step {}: NLL = {}".format(i, neg_log_likelihood))
  print("Final NLL = {}".format(neg_log_likelihood))
  
  return theta.numpy(), amp.numpy(), ls.numpy(), sigma.numpy()


def gp_smooth_loc_data(loc_data, target_var, n_iter=20000):
  t = (loc_data.date - loc_data.date.values[0]).dt.days.astype('float64')
  y = loc_data[target_var].values.astype('float64')
  return gp_smooth(t=tf.constant(t),
                   y=tf.constant(y),
                   n_iter=n_iter)
