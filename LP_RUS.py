from collections import defaultdict
import numpy as np
import random

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


def LP_RUS(y, percentage):
    samples_to_delete = int(y.shape[0] * percentage / 100)

    lp = LabelPowerset()
    labelsets = np.array(lp.transform(y))
    label_set_bags = defaultdict(list)

    for idx, label in enumerate(labelsets):
        label_set_bags[label].append(idx)

    mean_size = 0
    for label, samples in label_set_bags.items():
        mean_size += len(samples)
    
    mean_size //= len(label_set_bags)

    majority_bag = []
    for label, samples in label_set_bags.items():
        if len(samples) > mean_size:
            majority_bag.append(label)

    if len(majority_bag) == 0:
        print('There are no labels above the mean size. mean_size: ', mean_size)
        return []

    mean_reduction = samples_to_delete // len(majority_bag)


    def custom_sort(label):
        return len(label_set_bags[label])

    majority_bag.sort(key=custom_sort)
    acc_remainders = np.zeros(len(majority_bag), dtype=np.int32)
    del_samples = []

    for idx, label in enumerate(majority_bag):
        reduction_bag = min(len(label_set_bags[label]) - mean_size, mean_reduction)

        remainder = mean_reduction - reduction_bag

        if remainder == 0:
            extra_reduction = min(len(label_set_bags[label]) - reduction_bag - mean_size, acc_remainders[idx])
            reduction_bag += extra_reduction
            remainder = acc_remainders[idx] - extra_reduction

        distribute_remainder(remainder, acc_remainders, idx + 1)

        for i in range(reduction_bag):
            x = random.randint(0, len(label_set_bags[label]) - 1)
            del_samples.append(label_set_bags[label][x])
            del label_set_bags[label][x]

    return del_samples


# Example of usage
x, y = make_multilabel_classification(n_samples=1000, n_features=10, n_classes=8)

print('Positive samples per class:')
print(np.sum(y, axis=0))

# Send the labels and the percentage to remove
delete_idxs = LP_RUS(y, 25)

print('Samples to delete (count): ')
print(len(delete_idxs))

print('Positive samples to delete per class: ')
print(np.sum(y[delete_idxs, :], axis=0))
