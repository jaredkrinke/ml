import numpy as np
from code_for_hw3_part2 import raw, standard, one_hot, load_auto_data, perceptron, averaged_perceptron, xval_learning_alg, auto_data_and_labels

auto_data = load_auto_data("auto-mpg.tsv")

feature_sets = []
feature_sets.append([("cylinders", raw), ("displacement", raw), ("horsepower", raw), ("weight", raw), ("acceleration", raw), ("origin", raw)])
feature_sets.append([("cylinders", one_hot), ("displacement", standard), ("horsepower", standard), ("weight", standard), ("acceleration", standard), ("origin", one_hot)])
feature_sets.append([("cylinders", one_hot), ("horsepower", standard)])

def parameterized_learner(algorithm, T):
    return lambda data, labels: algorithm(data, labels, {"T": T})

# for T in [1, 10, 50]:
for T in [1]:
    for j, features in enumerate(feature_sets):
        print("[", end="")
        first = True
        # for i, algorithm in enumerate([perceptron, averaged_perceptron]):
        for algorithm in [averaged_perceptron]:
            learner = parameterized_learner(algorithm, T)
            data, labels = auto_data_and_labels(auto_data, features)
            result = xval_learning_alg(learner, data, labels, 10)
            # print(f"a={i}, T={T}, f={j}: {result}")
            print(result, end="")
            if first:
                print(", ", end="")
                first=False
        print("]")
    