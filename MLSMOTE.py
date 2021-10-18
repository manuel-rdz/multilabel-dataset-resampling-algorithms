import numpy as np
import random
import copy
import mld_metrics

from sklearn.datasets import make_multilabel_classification

def calculate_distance(a, b):
    return np.linalg.norm(a - b)


def distances_one_all(sample_idx, elements_idxs, x):
    distances = []
    for elem_idx in elements_idxs:
        distances.append((elem_idx, calculate_distance(x[sample_idx, :], x[elem_idx, :])))

    return distances

def new_sample(sample, ref_neighbor, neighbors, x, y):
    synth_sample = np.zeros(x.shape[1])

    for feature_idx in range(len(synth_sample)):
        # missing to add when feature is not numeric. In that case it should
        # put the most frequent value in the neighbors
        diff = x[ref_neighbor, feature_idx] - x[sample, feature_idx]
        offset = diff * random.random()
        value = x[sample, feature_idx] + offset

        synth_sample[feature_idx] = value

    labels_counts = y[sample, :]
    labels_counts = np.add(labels_counts, np.sum(y[neighbors, :], axis=0))
    labels = labels_counts > (len(neighbors) + 1) / 2

    return synth_sample, labels

def sortSamples(v):
    return v[1]

def MLSMOTE(x, y, k):
    mean_ir = mld_metrics.mean_ir(y)

    y_new = copy.deepcopy(y)
    x_new = copy.deepcopy(x)

    for label in range(y_new.shape[1]):
        ir_label = mld_metrics.ir_per_label(label, y_new)
        if ir_label > mean_ir:
            label_samples = y_new[:, label] == 1
            minority_bag = np.where(label_samples)[0]

            for sample_idx in minority_bag:
                distances = distances_one_all(sample_idx, minority_bag, x_new)
            
                distances.sort(key=sortSamples)

                # ignore the first one since it'll be sample
                neighbors = [v[0] for v in distances[1:k+1]]
                ref_neighbor = random.sample(neighbors, k=1)

                synth_sample, labels = new_sample(sample_idx, ref_neighbor, neighbors, x_new, y_new)

                y_new = np.append(y_new, [labels.astype(int)], axis=0)
                x_new = np.append(x_new, [synth_sample], axis=0)
    
    return x_new[x.shape[0]:], y_new[y.shape[0]:]


# Example of usage
x, y = make_multilabel_classification(n_samples=1000, n_features=10, n_classes=8)

print('Positive samples per class:')
print(np.sum(y, axis=0))

# Send the labels and the percentage to delete
nx, ny = MLSMOTE(x, y, 5)

print('Synthetic samples generated (count): ')
print(ny.shape[0])

print('Positive samples generated per class: ')
print(np.sum(ny, axis=0))
