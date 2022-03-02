import numpy as np
import code_for_hw3_part1 as lib

data_raw = np.array([[200, 800, 200, 800], [0.2,  0.2,  0.8,  0.8]])
labels = np.array([[-1, -1, 1, 1]])

# Transform
data = np.concatenate([data_raw, np.ones([1, data_raw.shape[1]])])

theta = np.array([0, 1, -0.5])

def margin(data, labels, theta):
    return labels.T * (theta.T @ data) / np.linalg.norm(theta)

print("Margin:", margin(data, labels, theta))
print("Margin 2:", margin(np.array([[0.001, 0.001, 1]]).T * data, labels, np.array([0, 1, -0.0005])))
print("Margin 3:", margin(np.array([[0.001, 1, 1]]).T * data, labels, theta))

print(np.array([[0.001, 1, 1]]).T * data)
print(np.linalg.norm(np.array([[0.001, 1, 1]]).T * data, axis=0))

data = np.array([[0.001, 1, 1]]).T * data

mistakes = 0
def incr(_):
    global mistakes
    mistakes += 1

def perceptron_through_origin(data, labels, params = {}, hook = None):
    T = params.get('T', 100)
    (d, n) = data.shape
    m = 0
    theta = np.zeros((d, 1))
    for t in range(T):
        all_good = True
        for i in range(n):
            x = data[:,i:i+1]
            y = labels[:,i:i+1]
            if y * lib.positive(x, theta, 0) <= 0.0:
                m += 1
                theta = theta + y * x
                all_good = False
                if hook: hook((theta))
        if all_good: break
    return theta

theta = perceptron_through_origin(data, labels, params={"T": 1000000}, hook=incr)
print(lib.score(data, labels, theta, np.array([[0]])))
print("Mistakes:", mistakes)
