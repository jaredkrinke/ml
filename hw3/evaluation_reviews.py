import numpy as np
from code_for_hw3_part2 import load_review_data, bag_of_words, extract_bow_feature_vectors, perceptron, averaged_perceptron, xval_learning_alg, reverse_dict

raw = load_review_data("reviews.tsv")
dictionary = bag_of_words([d.get("text") for d in raw])
fv = extract_bow_feature_vectors([d.get("text") for d in raw], dictionary)

def parameterized_learner(algorithm, T):
    return lambda data, labels: algorithm(data, labels, {"T": T})

# Best algorithm
# for T in [1, 10, 50]:
#     for algorithm in [perceptron, averaged_perceptron]:
#         learner = parameterized_learner(algorithm, T)
#         data = fv
#         labels = np.array([[d.get("sentiment") for d in raw]])
#         print(data.shape, labels.shape, end=" ")
#         print(xval_learning_alg(learner, data, labels, 10))

# Most important words
data = fv
labels = np.array([[d.get("sentiment") for d in raw]])
theta, theta_0 = parameterized_learner(averaged_perceptron, 10)(data, labels)
reverse_dictionary = reverse_dict(dictionary)

most_positive_indices = [x[0] for x in reversed(np.argsort(theta, axis=0))][:10]
print([reverse_dictionary[index] for index in most_positive_indices])

most_negative_indices = [x[0] for x in np.argsort(theta, axis=0)][:10]
print([reverse_dictionary[index] for index in most_negative_indices])

# Most extreme reviews
distances = theta.T @ data + theta_0
worst = [raw[index].get("text") for index in np.argsort(distances, axis=1)[0,:3]]
best = [raw[index].get("text") for index in np.argsort(distances, axis=1)[0,-3:]]
