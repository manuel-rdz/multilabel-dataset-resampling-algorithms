from collections import defaultdict
import numpy as np
import random
import math

from skmultilearn.problem_transform import LabelPowerset
from sklearn.datasets import make_multilabel_classification


def distribute_remainder(r, r_dist, idx):
    p = len(r_dist) - idx + 1
    value = r // p
    curr_rem = r % p

    r_dist[idx:] = np.add(r_dist[idx:], value)
    
    if curr_rem > 0:
        start = len(r_dist) - curr_rem
        r_dist[start:] = np.add(r_dist[start:], 1)


# return the idxs of the samples that have to be cloned
def LP_ROS(y, p):
    samples_to_clone = int(y.shape[0] * p / 100)

    lp = LabelPowerset()
    labelsets = np.array(lp.transform(y))
    label_set_bags = defaultdict(list)

    for idx, label in enumerate(labelsets):
        label_set_bags[label].append(idx)

    mean_size = 0
    for label, samples in label_set_bags.items():
        mean_size += len(samples)
    
    # ceiling
    mean_size = math.ceil(mean_size / len(label_set_bags))

    minority_bag = []
    for label, samples in label_set_bags.items():
        if len(samples) < mean_size:
            minority_bag.append(label)

    if len(minority_bag) == 0:
        print('There are no labels below the mean size. mean_size: ', mean_size)
        return []

    mean_increase = samples_to_clone // len(minority_bag)

    def custom_sort(label):
        return len(label_set_bags[label])

    minority_bag.sort(reverse=True, key=custom_sort)
    acc_remainders = np.zeros(len(minority_bag), dtype=np.int32)
    clone_samples = []

    for idx, label in enumerate(minority_bag):
        increase_bag = min(mean_size - len(label_set_bags[label]), mean_increase)

        remainder = mean_increase - increase_bag

        if remainder == 0:
            extra_increase = min(mean_size - len(label_set_bags[label]) - increase_bag, acc_remainders[idx])
            increase_bag += extra_increase
            remainder = acc_remainders[idx] - extra_increase

        distribute_remainder(remainder, acc_remainders, idx + 1)

        for i in range(increase_bag):
            x = random.randint(0, len(label_set_bags[label]) - 1)
            clone_samples.append(label_set_bags[label][x])

    return clone_samples


# Example of usage
x, y = make_multilabel_classification(n_samples=1000, n_features=10, n_classes=8)

print('Positive samples per class:')
print(np.sum(y, axis=0))

# Send the labels and the percentage to clone
clone_idxs = LP_ROS(y, 25)

print('Samples to clone (count): ')
print(len(clone_idxs))

print('Positive samples to clone per class: ')
print(np.sum(y[clone_idxs, :], axis=0))
