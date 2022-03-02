import numpy as np

# Fixed random seed
seed = 12345
rng = np.random.default_rng(seed)

# Labels
positive = 1
negative = -1

# Real data (CSV: param 1,2,...,4, label=0|1)
raw = np.loadtxt("data_banknote_authentication.txt", delimiter=",")
data = raw[:,:-1].T

print(data)

# # Switch negative labels from 0 to -1
# labels = raw[:,-1:].T
# labels = np.where(labels == 0, negative, labels)

# # Split into training and test data (50-50)
# sample_count = np.shapre(data)[1]
# training_data, test_data = np.split(data, )