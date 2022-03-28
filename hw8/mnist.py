# Not sure why this is needed for loading GPU DLLs... but GPU training is slower on this example anyway...
# import os
# os.add_dll_directory("C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.2/bin")
# os.add_dll_directory("C:/Program Files/NVIDIA/CUDNN/v8.3/bin")
# os.add_dll_directory("C:/Program Files/Microsoft Office/root/Office16/ODBC Drivers/Salesforce/lib") # zlibwapi...

from code_for_hw8_keras import run_keras_cnn_mnist, run_keras_fc_mnist, get_MNIST_data
from keras.layers import Conv1D, Conv2D, Dense, Dropout, Flatten, MaxPooling2D
from keras.initializers import VarianceScaling

def scale(tuple):
    train, validation = tuple
    return (train[0] / 256, train[1]), (validation[0] / 256, validation[1])

train, validation = scale(get_MNIST_data())

five_b = [Dense(input_dim=28*28, units=10, activation="softmax")]
# five_b = [Dense(input_dim=28*28, units=10, activation="softmax", kernel_initializer=VarianceScaling(scale=0.001, mode='fan_in', distribution='normal', seed=None))]

five_h = [Dense(input_dim=28*28, units=1024, activation="relu"), Dense(units=10, activation="softmax")]
five_i = [Dense(input_dim=28*28, units=512, activation="relu"), Dense(units=256, activation="relu"), Dense(units=10, activation="softmax")]

# run_keras_fc_mnist(train, validation, five_i, epochs=1, split=0.1, verbose=True)

five_j = [
    Conv2D(32, input_shape=(28,28,1), kernel_size=(3,3), activation="relu"),
    MaxPooling2D(pool_size=(2,2)),
    Conv2D(64, kernel_size=(3,3), activation="relu"),
    MaxPooling2D(pool_size=(2,2)),
    Flatten(),
    Dense(units=128, activation="relu"),
    Dropout(0.5),
    Dense(units=10, activation="softmax")
]

# run_keras_cnn_mnist(train, validation, five_j, epochs=1)

train_20, validation_20 = scale(get_MNIST_data(shift=20))
size = 48
five_ki = [
    Dense(input_dim=size*size, units=512, activation="relu"),
    Dense(units=256, activation="relu"),
    Dense(units=10, activation="softmax")
]

five_kj = [
    Conv2D(32, input_shape=(size,size,1), kernel_size=(3,3), activation="relu"),
    MaxPooling2D(pool_size=(2,2)),
    Conv2D(64, kernel_size=(3,3), activation="relu"),
    MaxPooling2D(pool_size=(2,2)),
    Flatten(),
    Dense(units=128, activation="relu"),
    Dropout(0.5),
    Dense(units=10, activation="softmax")
]

run_keras_fc_mnist(train_20, validation_20, five_ki, epochs=1, split=0.1, verbose=True)
run_keras_cnn_mnist(train_20, validation_20, five_kj, epochs=1, split=0.1, verbose=True)
