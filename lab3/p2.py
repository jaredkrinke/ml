import numpy as np
import csv
from shared import perceptron, eval_classifier

def bag_of_words(values):
    # NumPy way to do this?
    rows = np.char.split(values, sep=" ")
    words = np.unique(np.concatenate(rows))
    word_to_index = {words[i]: i for i in range(0, np.shape(words)[0])}
    result = np.zeros((rows.shape[0], words.shape[0]))
    for j, row in enumerate(rows):
        row = rows[j]
        for word in row:
            result[j][word_to_index[word]] = 1
    return result

# sentiment	productId	userId	summary	text	helpfulY	helpfulN
rng = np.random.default_rng(12345)
# TODO: invalid_raise?
raw =  np.genfromtxt("reviews.tsv", delimiter="\t", usecols=(0,3,4), encoding="utf-8", dtype=None, names=True, invalid_raise=False)
rng.shuffle(raw)

# Use (separate) bag of words encoding for summary and text
def prepare(data):
    return np.concatenate([bag_of_words(data[:]["summary"]), bag_of_words(data[:]["text"])], axis=1), data[:]["sentiment"].reshape([-1, 1])

data, labels = prepare(raw)
n = (np.shape(data)[0] * 9) // 10
data_train, data_test = data[:n], data[n:]
labels_train, labels_test = labels[:n], labels[n:]

print(f"{data.shape[0]} rows, {data.shape[1]} columns...")

print(eval_classifier(perceptron, data_train.T, labels_train.T, data_test.T, labels_test.T))
