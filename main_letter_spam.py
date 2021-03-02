'''Main function for UCI letter and spam datasets.
'''

# Necessary packages
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse

import numpy as np
from sklearn.model_selection import KFold

from data_loader import data_loader
from gain import gain


def main(args):
    '''Main function for UCI letter and spam datasets.

    Args:
      - data_name: letter or spam
      - miss_rate: probability of missing components
      - batch:size: batch size
      - hint_rate: hint rate
      - alpha: hyperparameter
      - iterations: iterations

    Returns:
      - imputed_data_x: imputed data
      - rmse: Root Mean Squared Error
    '''

    data_name = args.data_name
    miss_rate = args.miss_rate

    gain_parameters = {'batch_size': args.batch_size,
                       'hint_rate': args.hint_rate,
                       'alpha': args.alpha,
                       'iterations': args.iterations}

    # Load data and introduce missingness
    ori_data_x, miss_data_x, data_m = data_loader(data_name, miss_rate, args.mechanism)

    # Impute missing data
    if args.kfold:
        rmse_list = []
        for i, (train_index, test_index) in enumerate(KFold(shuffle=True, random_state=1).split(ori_data_x)):
            rmse = gain(miss_data_x, gain_parameters, ori_data_x, train_index, test_index, args.mechanism)
            rmse_list.append(rmse)
        print(np.mean(rmse_list), np.std(rmse_list))
    else:
        train_index = test_index = range(len(ori_data_x))
        gain(miss_data_x, gain_parameters, ori_data_x, train_index, test_index, args.mechanism)


if __name__ == '__main__':
    # Inputs for the main function
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--data_name',
        choices=['letter', 'spam', 'breast', 'news', 'credit'],
        default='spam',
        type=str)
    parser.add_argument(
        '--mechanism',
        choices=['mcar', 'mar'],
        default='mcar',
        type=str)
    parser.add_argument(
        '--miss_rate',
        help='missing data probability',
        default=0.2,
        type=float)
    parser.add_argument(
        '--batch_size',
        help='the number of samples in mini-batch',
        default=128,
        type=int)
    parser.add_argument(
        '--hint_rate',
        help='hint probability',
        default=0.9,
        type=float)
    parser.add_argument(
        '--alpha',
        help='hyperparameter',
        default=100,
        type=float)
    parser.add_argument(
        '--iterations',
        help='number of training interations',
        default=10000,
        type=int)
    parser.add_argument(
        '--kfold',
        help='K-Fold',
        action='store_true'
    )
    args = parser.parse_args()

    # Calls main function
    main(args)
