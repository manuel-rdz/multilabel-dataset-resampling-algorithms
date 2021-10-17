import numpy as np
import random
import copy
import mld_metrics

from sklearn.datasets import make_multilabel_classification


def ML_ROS(y, p):
    y_new = copy.deepcopy(y)

    samples_to_clone = int(y.shape[0] * p / 100)
    samples_per_label = {}
    samples = np.arange(y.shape[0])

    for label in range(y.shape[1]):
        label_samples = y[:, label] == 1
        samples_per_label[label] = samples[label_samples]

    mean_ir = mld_metrics.mean_ir(y)
    minority_bag = []

    for i in range(y.shape[1]):
        if mld_metrics.ir_per_label(i, y) > mean_ir:
            minority_bag.append(i)
    
    clone_samples = []
    while samples_to_clone > 0 and len(minority_bag) > 0:
        for label in minority_bag:
            x = random.randint(0, len(samples_per_label[label]) - 1)
            y_new = np.append(y_new, [y[samples_per_label[label][x]]], axis=0)

            if mld_metrics.ir_per_label(label, y_new) <= mean_ir:
                minority_bag.remove(label)

            clone_samples.append(samples_per_label[label][x])
            samples_to_clone -= 1

    return clone_samples


# Example of usage
x, y = make_multilabel_classification(n_samples=1000, n_features=10, n_classes=8)

print('Positive samples per class:')
print(np.sum(y, axis=0))

# Send the labels and the percentage to clone
clone_idxs = ML_ROS(y, 25)

print('Samples to clone (count): ')
print(len(clone_idxs))

print('Positive samples to clone per class: ')
print(np.sum(y[clone_idxs, :], axis=0))
