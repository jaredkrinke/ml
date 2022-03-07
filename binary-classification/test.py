import numpy as np
import binary_classification as lib

# Fixed random seed
seed = 12345
rng = np.random.default_rng(seed)

# Labels
positive = 1
negative = -1

# Real data (CSV: param 1,2,...,4, label=0|1)
samples =  np.genfromtxt("data_banknote_authentication.txt", delimiter=",", skip_header=True)

# Shuffle and normalize negative labels to -1 instead of 0
rng.shuffle(samples)
data_all = samples[:,:-1].T
labels_all = np.where(samples[:,-1:] == 0, negative, samples[:,-1:]).T

# Split data (roughly) 50-50
[data_training, data_test] = np.array_split(data_all, 2, axis=1)
[labels_training, labels_test] = np.array_split(labels_all, 2, axis=1)

# Trivial model
def train_trivial(_data, _labels):
    return np.array([[.1], [.2], [.3], [.4]]), np.array([[.5]])

# Compare
def format_model(normal, offset):
    def format_float(f): return "{0:1.3f}".format(f)
    return ", ".join(map(format_float, normal.reshape([-1]))) + "; " + format_float(offset[0,0])

print("ETrain\tETest\tECross\tStrategy")
for model in [train_trivial, lib.train_random, lib.train_perceptron, lib.train_support_vector_machine]:
    normal, offset = model(data_training, labels_training)
    error_training = lib.score_accuracy(data_training, labels_training, normal, offset)
    error_test = lib.score_accuracy(data_test, labels_test, normal, offset)
    print("{0:1.4f}\t{1:1.4f}\t{2:1.4f}\t{3}:\t{4}".format(error_training, error_test, -1, model.__name__, format_model(normal, offset)))

# TODO: Cross-fold validation