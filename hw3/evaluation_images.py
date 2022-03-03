import numpy as np
from code_for_hw3_part2 import load_mnist_data, get_classification_accuracy

for [a, b] in [[0, 1], [2, 4], [6, 8], [9, 0]]:
    digit_to_data = load_mnist_data([a, b])
    # TODO: Stop switching back and forth between Python lists and NumPy arrays
    data = np.zeros([28 * 28, 0])
    labels = []
    for digit in [a, b]:
        digit_data = digit_to_data[digit]
        for image in digit_data["images"]:
            data = np.append(data, np.array(image).reshape([28 * 28, 1]), axis=1)
            labels.append(1 if digit == a else -1)

    labels = np.array(labels).reshape(1, -1)    
    print(get_classification_accuracy(data, labels))
