import numpy as np

# Linear models for binary classification
def classify_binary(data, normal, offset):
    return np.sign(((data.T @ normal) + offset))

# Objective functions
def objective_negative_log_likelihood(data, labels, normal, offset):
    pass

def objective_support_vector_machine(data, lables, normal, offset):
    pass

# Scoring
def score_correct(data, labels, normal, offset):
    predicted = classify_binary(data, normal, offset)
    return np.sum(predicted.T == labels, axis=1, keepdims=True)

def score_accuracy(data, labels, normal, offset):
    return np.mean(classify_binary(data, normal, offset) == labels, axis=1)[0]

# Learning algorithms for linear classifiers
def train_random(data, labels, iterations=1000, seed=12345):
    dimensions = data.shape[0]
    rng = np.random.default_rng(seed)
    normals = rng.uniform(-1, 1, size=(dimensions, iterations))
    offsets = rng.uniform(-1, 1, size=(1, iterations))
    index_best = np.argmax(score_correct(data, labels, normals, offsets))
    return normals[:,index_best].reshape([-1, 1]), offsets[:,index_best].reshape([1, 1])

def train_perceptron(data, labels, passes=50):
    dimensions, samples = data.shape
    normal = np.zeros([dimensions, 1])
    offset = np.zeros([1, 1])
    for _ in range(0, passes):
        all_good = True
        # TODO: NumPy-ify this loop
        for i in range(0, samples):
            point = data[:,i:i+1]
            label = labels[:,i:i+1]
            if label * classify_binary(point, normal, offset) <= 0.0:
                normal = normal + label * point
                offset = offset + label
                all_good = False
        if all_good: break
    return normal, offset

def train_gradient_descent(data, labels, loss_function, loss_function_gradient=None, regularizer_coefficient=0.01):
    pass
