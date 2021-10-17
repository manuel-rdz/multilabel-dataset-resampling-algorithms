import numpy as np
import math

from sklearn.datasets import make_multilabel_classification


# imbalance rate per label
def ir_per_label(label, y):
    a = np.sum(y[:, label], axis=0)
    b = np.max(np.sum(y, axis=0))

    return b / a


# mean imbalance rate
def mean_ir(y):
    mean = 0.0

    for i in range(y.shape[1]):
        mean += ir_per_label(i, y)
    
    return mean / y.shape[1]


# used to calculate cvir
def ir_per_label_alpha(y, mean_ir_val):
    mean = 0.0

    for i in range(y.shape[1]):
        mean += ((ir_per_label(i, y) - mean_ir_val) ** 2) / (y.shape[1] - 1)
    
    return math.sqrt(mean)


# coefficient of variation of IRperLabel
def cvir(y):
    mean_ir_val = mean_ir(y)
    alpha = ir_per_label_alpha(y, mean_ir_val)

    return alpha / mean_ir_val



# Example of usage
x, y = make_multilabel_classification(n_samples=1000, n_features=10, n_classes=8)

print('Positive samples per class:')
print(np.sum(y, axis=0))

irs = []

for i in range(y.shape[1]):
    irs.append(ir_per_label(i, y))

print('Imbalance Rate per class:')
print(irs)

print('Mean Imbalance Rate: ')
print(mean_ir(y))

print('Coefficient of variation per label:')
print(cvir(y))
