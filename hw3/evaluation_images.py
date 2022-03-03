import numpy as np
from code_for_hw3_part2 import load_mnist_data, get_classification_accuracy

def digit_pair(a, b):
    digit_to_data = load_mnist_data([a, b])
    data = []
    labels = []
    for digit in [a, b]:
        digit_data = digit_to_data[digit]
        data += digit_data["images"]
        labels += [1 if d == a else -1 for d in digit_data["labels"][0]]
    return data, np.array([labels])

for [a, b] in [[0, 1], [2, 4], [6, 8], [9, 0]]:
    raw_data, labels = digit_pair(a, b)
    data = np.zeros([28 * 28, 0])
    for image in raw_data:
        data = np.append(data, np.array(image).reshape([28 * 28, 1]), axis=1)
    print(get_classification_accuracy(data, labels))

def row_average_features(x): return np.mean(x, axis=1, keepdims=True)
def col_average_features(x): return np.mean(x, axis=0).reshape(-1, 1)
def top_bottom_features(x): return np.array([[np.mean(x[:x.shape[0]//2])], [np.mean(x[x.shape[0]//2:])]])

for [a, b] in [[0, 1], [2, 4], [6, 8], [9, 0]]:
    raw_data, labels = digit_pair(a, b)
    results = []
    for f in [row_average_features, col_average_features, top_bottom_features]:
        data = np.concatenate([f(x) for x in raw_data], axis=1)
        results.append("{0:0.03f}".format(get_classification_accuracy(data, labels)))
    print(f"[{', '.join(results)}]")
