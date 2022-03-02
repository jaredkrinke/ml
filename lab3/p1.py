import numpy as np
import sys

# TODO: There must be some better way to import from a sibling directory...
sys.path.append("..")
from hw2.hw2 import eval_classifier, perceptron, xval_learning_alg

rng = np.random.default_rng(12345)

raw = np.loadtxt("auto-mpg.tsv", delimiter="\t", skiprows=1, usecols=range(0, 8))
rng.shuffle(raw)
labels = raw[:,[0]]
data_raw = raw[:,1:]

def noop(values):
    return values

def standardize(values):
    mean = np.mean(values)
    stdev = np.std(values)
    return (values - mean) / stdev

def one_hot(values):
    unique, inverse = np.unique(values, return_inverse=True)
    return np.eye(unique.shape[0])[inverse]

# cylinders	displacement	horsepower	weight	acceleration	model_year	origin	car_name
# 8	304	193	4732	18.5	70	1	"hi 1200d"

def fr0(data):
    # All numbers, as is
    return data

def fr1(data):
    # Standardize all numbers
    column_count = np.shape(data)[1]
    return np.concatenate([standardize(column) for column in np.split(data, column_count, axis=1)], axis=1)

def fr2(data):
    # Use one-hot encoding for model_year, origin and standardize all the rest
    column_count = np.shape(data)[1]
    columns = np.split(data, column_count, axis=1)
    return np.concatenate([standardize(columns[0]), standardize(columns[1]), standardize(columns[2]), standardize(columns[3]), standardize(columns[4]), one_hot(columns[5]), one_hot(columns[6])], axis=1)

data_sets = [fr0(data_raw), fr1(data_raw), fr2(data_raw)]

print("Strat.\tETrain\tETest\tECross")
for i, data in enumerate(data_sets):
    n = (np.shape(data)[0] * 9) // 10
    data_train, data_test = data[:n], data[n:]
    labels_train, labels_test = labels[:n], labels[n:]
    e_train = eval_classifier(perceptron, data.T, labels.T, data.T, labels.T)
    e_test = eval_classifier(perceptron, data_train.T, labels_train.T, data_test.T, labels_test.T)
    e_cross = xval_learning_alg(perceptron, data.T, labels.T, 10)
    print("{0:1d}\t{1:1.4f}\t{2:1.4f}\t{3:1.4f}".format(i, e_train, e_test, e_cross))
