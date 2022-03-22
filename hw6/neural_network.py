import numpy as np

def rectified_linear_unit(x):
    return np.maximum(np.zeros(x.shape), x)

def soft_max(x):
    return np.exp(x) / np.sum(np.exp(x))

def create_neural_network(inputs, outputs, hidden_layer_units, hidden_layers, activation_hidden, activation_final, weights_initial, offsets_initial):
    assert(len(weights_initial) == len(offsets_initial))
    for layer, weights in enumerate(weights_initial):
        assert(weights.shape[0] == inputs if layer == 0 else hidden_layer_units)
        assert(weights.shape[1] == outputs if layer == hidden_layers else hidden_layer_units)

    for layer, offsets in enumerate(offsets_initial):
        assert(offsets.shape[0] == outputs if layer == hidden_layers else hidden_layer_units)
        assert(offsets.shape[1] == 1)

    def neural_network(input):
        result = input
        assert(input.shape[0] == inputs)
        assert(input.shape[1] == 1)
        for layer in range(hidden_layers + 1):
            weights = weights_initial[layer]
            offsets = offsets_initial[layer]
            result = weights.T @ result + offsets
            result = activation_hidden(result) if layer < hidden_layers else activation_final(result)
        assert(result.shape[0] == outputs)
        assert(result.shape[1] == 1)
        return result
    return neural_network

layer1_weights = np.array([[1, 0, -1, 0], [0, 1, 0, -1]])
layer1_offsets = np.array([[-1, -1, -1, -1]]).T

layer2_weights = np.array([[1, -1], [1, -1], [1, -1], [1, -1]])
layer2_offsets = np.array([[0, 2]]).T

nn = create_neural_network(2, 2, 4, 1, rectified_linear_unit, soft_max, [layer1_weights, layer2_weights], [layer1_offsets, layer2_offsets])
print(nn(np.array([[3, 14]]).T))

# print(soft_max(np.array([[-1, 0, 1]]).T))
