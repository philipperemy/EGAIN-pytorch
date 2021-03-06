# Generative Adversarial Imputation Networks (GAIN)

Pytorch implementation of EGAIN, an extension of GAIN.

Refer to this [link](https://github.com/jsyoon0823/GAIN) for the official implementation in Tensorflow V1.


This directory contains implementations of GAIN framework for imputation
using two UCI datasets.

-   UCI Letter (https://archive.ics.uci.edu/ml/datasets/Letter+Recognition)
-   UCI Spam (https://archive.ics.uci.edu/ml/datasets/Spambase)

### Command inputs

-   data_name: letter or spam
-   miss_rate: probability of missing components
-   batch_size: batch size
-   hint_rate: hint rate
-   alpha: hyperparameter
-   iterations: iterations

### Example command

```shell
$ python3 main_letter_spam.py --data_name spam --miss_rate 0.2 --batch_size 128 --hint_rate 0.9 --alpha 100 --iterations 10000
```

