import numpy as np

# Generic helpers
def rv(values):
    return np.array([values])

def cv(values):
    return rv(values).T

norm = np.linalg.norm

# Classification helpers
# TODO: how to convey dimensions on all these arguments and return values?
def classify(data, ths, th0s):
    return np.sign(((data.T @ ths) + th0s)).T

# TODO: Custom loss function?
def evaluate(data, labels, ths, th0s):
    return np.sum(classify(data, ths, th0s) == labels, axis=1, keepdims=True)

def find_best(data, labels, ths, th0s):
    # data: a d by n array of floats (representing n data points in d dimensions)
    # labels: a 1 by n array of elements in (+1, -1), representing target labels
    # ths: a d by m array of floats representing m candidate hyperplane normals
    # th0s: a 1 by m array of the corresponding m candidate hyperplane offsets
    best_index = np.argmax(evaluate(data, labels, ths, th0s))
    return cv(ths[:,best_index]), cv(th0s[:,best_index])

## Test data (from )
# data = np.array([[1, 1, 2, 1, 2], [2, 3, 1, -1, -1]])
# labels = np.array([[-1, -1, 1, 1, 1]])
# ths = np.array([[0.98645534, -0.02061321, -0.30421124, -0.62960452, 0.61617711, 0.17344772, -0.21804797, 0.26093651, 0.47179699, 0.32548657], [0.87953335, 0.39605039, -0.1105264, 0.71212565, -0.39195678, 0.00999743, -0.88220145, -0.73546501, -0.7769778, -0.83807759]])
# th0s = np.array([[0.65043158, 0.61626967, 0.84632592, -0.43047804, -0.91768579, -0.3214327, 0.0682113, -0.20678004, -0.33963784, 0.74308104]])

# # Expected output: (theta, theta_0)
# # [[[0.32548657], [-0.83807759]], [[0.74308104]]]
# # print(find_best(data, labels, ths, th0s))

# print("[[[0.32548657], [-0.83807759]], [[0.74308104]]]")
# print([x.tolist() for x in find_best(data,labels, ths, th0s)])

## Real data (CSV: param 1,2,...,4, label)
# TODO: Split into training and test data sets
positive = 1
negative = -1

raw = np.loadtxt("data_banknote_authentication.txt", delimiter=",")
data = raw[:,:-1].T

# Set negative labels to -1
labels = raw[:,-1:].T
labels = np.where(labels == 0, -1, labels)

# Random model generation
rng = np.random.default_rng(12345)
# TODO: something more robust
min = -1
max = 1
# min = -1000000000
# max = 1000000000
dimensions = np.shape(data)[0]

# TODO: Generate and evaluate in batches
def generate_random_model(data):
    thetas = rng.uniform(-1, 1, size=(dimensions, 1))
    theta_0s = rng.uniform(min, max, size=(1, 1))
    return thetas, theta_0s

def stats(data, labels, theta, theta_0):
    classifications = classify(data, theta, theta_0)
    count = np.shape(labels)[1]
    # TODO: Can this be done with np.choose?
    positive_label_count = np.sum(labels == positive)
    negative_label_count = np.sum(labels == negative)
    assert(count == positive_label_count + negative_label_count)
    true_positive_count = np.sum(np.logical_and(classifications == labels, labels == positive))
    true_negative_count = np.sum(np.logical_and(classifications == labels, labels == negative))
    false_positive_count = np.sum(np.logical_and(classifications != labels, labels == positive))
    false_negative_count = np.sum(np.logical_and(classifications != labels, labels == negative))
    assert(count == sum([true_positive_count, true_negative_count, false_positive_count, false_negative_count]))
    return {
        "accuracy": (true_positive_count + true_negative_count) / count,
        "false_positive_rate": false_positive_count / positive_label_count,
        "false_negative_rate": false_negative_count / negative_label_count,
    }

# TODO: See note on rewriting above
thetas = np.zeros((dimensions, 0))
theta_0s = np.zeros((1, 0))
def run(count):
    global thetas
    global theta_0s
    
    new_theta, new_theta_0 = generate_random_model(data)
    thetas = np.append(thetas, new_theta, axis=1)
    theta_0s = np.append(theta_0s, new_theta_0, axis=1)
    theta, theta_0 = find_best(data, labels, thetas, theta_0s)

    return stats(data, labels, theta, theta_0)

last_s = None
for x in range(1, 100):
    s = run(x)
    if last_s == None or (last_s["accuracy"] < s["accuracy"]):
        print(x, s)
        last_s = s
