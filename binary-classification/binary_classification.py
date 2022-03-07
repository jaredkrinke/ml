import numpy as np

# Linear models for binary classification
def classify_binary(data, normal, offset):
    return np.sign(((data.T @ normal) + offset))

# Objective functions
def objective_negative_log_likelihood(data, labels, normal, offset):
    pass

def objective_support_vector_machine(x, y, normal, offset, regularizer_coefficient):
    def hinge(v): return np.where(v < 1, 1 - v, 0)
    hinge_loss = hinge(y * (normal.T @ x + offset))
    return 1 / x.shape[1] * np.sum(hinge_loss) + regularizer_coefficient * np.linalg.norm(normal) ** 2

# Objective gradients
def gradient_objective_support_vector_machine(x, y, normal, offset, regularizer_coefficient):
    def d_hinge(v): return np.where(v < 1, -1, 0)
    d_hinge_d_normal = d_hinge(y * (normal.T @ x + offset)) * x * y
    d_objective_d_normal = np.mean(d_hinge_d_normal, axis=1, keepdims=True) + 2 * regularizer_coefficient * normal
    d_hinge_d_offset = d_hinge(y * (normal.T @ x + offset)) * y
    d_objective_d_offset = np.mean(d_hinge_d_offset, axis=1, keepdims=True)
    return np.vstack((d_objective_d_normal, d_objective_d_offset))

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

def descend_gradient(df, x0, step_size_fn, max_iter):
    x = x0.copy()
    for i in range(0, max_iter):
        x -= df(x) * step_size_fn(i)
    return x

def train_gradient_descent(data, labels, gradient_objective, regularizer_coefficient=0.01):
    normal = np.zeros([data.shape[0], 1])
    offset = 0
    x0 = np.vstack((normal, [[offset]]))

    extract_normal = lambda x: x[:-1]
    extract_offset = lambda x: x[-1:]

    def df(x): return gradient_objective(data, labels, extract_normal(x), extract_offset(x), regularizer_coefficient)
    def svm_min_step_size_fn(i): return 2/(i+1)**0.5
    
    x = descend_gradient(df, x0, svm_min_step_size_fn, 50)
    return extract_normal(x), extract_offset(x)

def train_support_vector_machine(data, labels, regularizer_coefficient=0.01):
    return train_gradient_descent(data, labels, gradient_objective_support_vector_machine, regularizer_coefficient)