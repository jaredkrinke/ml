import code_for_hw8_keras as hw8
import numpy as np

# architectures = hw8.archs(3)
# for i, architecture in enumerate(architectures):
#     print(f"Architecture {i}")
#     hw8.run_keras_2d("3class", architecture, 10, trials=1, display=False)

from keras.layers import Conv1D, Conv2D, Dense, Dropout, Flatten, MaxPooling2D
# architecture = [Dense(input_dim=2, units=100, activation='relu'), Dense(units=100, activation='relu'), Dense(units=2, activation="softmax")]
architecture = hw8.archs(3)[0]
hw8.run_keras_2d("3class", architecture, epochs=10, verbose=False, display=False)

W, W_0 = architecture[0].get_weights()

points = [[-1,0], [1,0], [0,-11], [0,1], [-1,-1], [-1,1], [1,1], [1,-1]]
def soft_max(x): return np.exp(x) / np.sum(np.exp(x))
for x in points:
    print(f"{x}: {np.argmax(soft_max(W.T@x + W_0))}")
