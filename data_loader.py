'''Data loader for UCI letter, spam and MNIST datasets.
'''

# Necessary packages
import numpy as np
from keras.datasets import mnist
from sklearn.datasets import load_breast_cancer

from utils import binary_sampler


def data_loader(data_name, miss_rate):
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

    # Introduce missing data
    data_m = binary_sampler(1 - miss_rate, no, dim)
    miss_data_x = data_x.copy()
    miss_data_x[data_m == 0] = np.nan

    return data_x, miss_data_x, data_m
