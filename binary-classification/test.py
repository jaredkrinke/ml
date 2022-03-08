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
data = samples[:,:-1].T
labels = np.where(samples[:,-1:] == 0, negative, samples[:,-1:]).T

# Compare
def format_model(normal, offset):
    def format_float(f): return "{0:1.3f}".format(f)
    return ", ".join(map(format_float, normal.reshape([-1]))) + "; " + format_float(offset[0,0])

def train_and_evaluate(train, data_training, labels_training, data_test, labels_test):
    normal, offset = train(data_training, labels_training)
    return lib.score_accuracy(data_test, labels_test, normal, offset)

def evaluate_cross_fold(train, data, labels, k):
    data_chunks = np.array_split(data, k, axis=1)
    label_chunks = np.array_split(labels, k, axis=1)
    score_sum = 0
    for i in range(k):
        data_train = np.concatenate(data_chunks[:i] + data_chunks[i+1:], axis=1)
        labels_train = np.concatenate(label_chunks[:i] + label_chunks[i+1:], axis=1)
        data_test = np.array(data_chunks[i])
        labels_test = np.array(label_chunks[i])
        score_sum += train_and_evaluate(train, data_train, labels_train, data_test, labels_test)
    return score_sum/k

print("ECross\tStrategy")
for train in [lib.train_random, lib.train_perceptron, lib.train_support_vector_machine]:
    # normal, offset = model(data_training, labels_training)
    # error_training = lib.score_accuracy(data_training, labels_training, normal, offset)
    # error_test = lib.score_accuracy(data_test, labels_test, normal, offset)
    # print("{0:1.4f}\t{1:1.4f}\t{2:1.4f}\t{3}:\t{4}".format(error_training, error_test, -1, model.__name__, format_model(normal, offset)))
    error_cross = evaluate_cross_fold(train, data, labels, 10)
    print("{0:1.4f}\t{1}".format(error_cross, train.__name__))
