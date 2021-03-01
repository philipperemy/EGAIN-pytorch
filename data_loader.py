'''Data loader for UCI letter, spam and MNIST datasets.
'''

import numpy as np
import torch
# Necessary packages
from keras.datasets import mnist
from scipy import optimize
from sklearn.datasets import load_breast_cancer

from utils import binary_sampler


def pick_coeffs(X, idxs_obs=None, idxs_nas=None, self_mask=False):
    n, d = X.shape
    if self_mask:
        coeffs = torch.randn(d)
        Wx = X * coeffs
        coeffs /= torch.std(Wx, 0)
    else:
        d_obs = len(idxs_obs)
        d_na = len(idxs_nas)
        coeffs = torch.randn(d_obs, d_na)
        Wx = X[:, idxs_obs].mm(coeffs)
        coeffs /= torch.std(Wx, 0, keepdim=True)
    return coeffs


def fit_intercepts(X, coeffs, p, self_mask=False):
    if self_mask:
        d = len(coeffs)
        intercepts = torch.zeros(d)
        for j in range(d):
            def f(x):
                return torch.sigmoid(X * coeffs[j] + x).mean().item() - p

            intercepts[j] = optimize.bisect(f, -50, 50)
    else:
        d_obs, d_na = coeffs.shape
        intercepts = torch.zeros(d_na)
        for j in range(d_na):
            def f(x):
                return torch.sigmoid(X.mv(coeffs[:, j]) + x).mean().item() - p

            intercepts[j] = optimize.bisect(f, -50, 50)
    return intercepts


def MAR_mask(X, p, p_obs):
    """
    Missing at random mechanism with a logistic masking model. First, a subset of variables with *no* missing values is
    randomly selected. The remaining variables have missing values according to a logistic model with random weights,
    re-scaled so as to attain the desired proportion of missing values on those variables.

    Parameters
    ----------
    X : torch.DoubleTensor or np.ndarray, shape (n, d)
        Data for which missing values will be simulated. If a numpy array is provided,
        it will be converted to a pytorch tensor.

    p : float
        Proportion of missing values to generate for variables which will have missing values.

    p_obs : float
        Proportion of variables with *no* missing values that will be used for the logistic masking model.

    Returns
    -------
    mask : torch.BoolTensor or np.ndarray (depending on type of X)
        Mask of generated missing values (True if the value is missing).

    """

    n, d = X.shape

    to_torch = torch.is_tensor(X)  ## output a pytorch tensor, or a numpy array
    if not to_torch:
        X = torch.from_numpy(X)

    mask = torch.zeros(n, d).bool() if to_torch else np.zeros((n, d)).astype(bool)

    d_obs = max(int(p_obs * d), 1)  ## number of variables that will have no missing values (at least one variable)
    d_na = d - d_obs  ## number of variables that will have missing values

    ### Sample variables that will all be observed, and those with missing values:
    idxs_obs = np.random.choice(d, d_obs, replace=False)
    idxs_nas = np.array([i for i in range(d) if i not in idxs_obs])

    ### Other variables will have NA proportions that depend on those observed variables, through a logistic model
    ### The parameters of this logistic model are random.

    ### Pick coefficients so that W^Tx has unit variance (avoids shrinking)
    coeffs = pick_coeffs(X, idxs_obs, idxs_nas)
    ### Pick the intercepts to have a desired amount of missing values
    intercepts = fit_intercepts(X[:, idxs_obs], coeffs, p)

    ps = torch.sigmoid(X[:, idxs_obs].mm(coeffs) + intercepts)

    ber = torch.rand(n, d_na)
    mask[:, idxs_nas] = ber < ps

    return mask


def data_loader(data_name, miss_rate, mechanism):
    '''Loads datasets and introduce missingness.

    Args:
      - data_name: letter, spam, or mnist
      - miss_rate: the probability of missing components

    Returns:
      data_x: original data
      miss_data_x: data with missing values
      data_m: indicator matrix for missing components
    '''

    # Load data
    if data_name in ['letter', 'spam']:
        file_name = 'data/' + data_name + '.csv'
        data_x = np.loadtxt(file_name, delimiter=",", skiprows=1)
    elif data_name == 'mnist':
        (data_x, _), _ = mnist.load_data()
        data_x = np.reshape(np.asarray(data_x), [60000, 28 * 28]).astype(float)
    elif data_name == 'breast':
        data_x = load_breast_cancer()['data']
    elif data_name == 'news':
        data_x = np.loadtxt('data/OnlineNewsPopularity1.csv', delimiter=",", skiprows=1)
    elif data_name == 'credit':
        data_x = np.loadtxt('data/default_of_credit_cards_clients.csv', delimiter=",", skiprows=2)
    else:
        raise Exception('Unknown dataset.')

    # Parameters
    no, dim = data_x.shape

    if mechanism == 'mcar':
        # Introduce missing data
        data_m = binary_sampler(1 - miss_rate, no, dim)
    else:
        data_m = 1 - MAR_mask(np.array(data_x, dtype=np.float32), p=miss_rate, p_obs=miss_rate)
    miss_data_x = data_x.copy()
    miss_data_x[data_m == 0] = np.nan

    return data_x, miss_data_x, data_m
