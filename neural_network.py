import json
import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def dsigmoid(y):
    return y * (1 - y)

def relu(x):
    return x * (x > 0)

def drelu(x):
    return 1. * (x > 0)
    
def tanh(x):
    return np.tanh(x)

def dtanh(y):
    return 1 - (y * y)

activations = {
    "sigmoid": (sigmoid, dsigmoid),
    "tanh": (tanh, dtanh),
    "relu": (relu, drelu),
}

def rs_np(r, c):
    return np.random.standard_normal((r,c))/c**.5
    
def r_np(r, c):
    return np.random.rand(r, c)

def a_np(a):
    return np.transpose(np.array(a, ndmin=2))

class NeuralNetwork:
    def __init__(self, input_nodes, output_nodes, lr=0.1, activation_funk="sigmoid"):
        self.layer_sizes = [input_nodes, output_nodes]
        self.weights = [rs_np(output_nodes, input_nodes)]
        self.biases = [r_np(output_nodes, 1)]
        self.learning_rate = lr
        self.activations = [activation_funk]

    @staticmethod
    def add_before_last(loc, n):
        loc.insert(len(loc)-1, n)

    def add_layer(self, neuron_layer, activation="sigmoid"):
        NeuralNetwork.add_before_last(self.layer_sizes, neuron_layer)
        NeuralNetwork.add_before_last(self.activations, activation)

        NeuralNetwork.add_before_last(self.biases, r_np(neuron_layer, 1))

        self.weights.clear()
        weight_shapes = zip(self.layer_sizes[1:], self.layer_sizes[:-1])
        for (row, col) in weight_shapes:
            self.weights.append(rs_np(row, col))

    def predict(self, input_array):
        before = a_np(input_array)
        for (a, w, b) in zip(self.activations, self.weights, self.biases):
            before = np.matmul(w, before) + b
            before = activations[a][0](before)

        return before.flatten().tolist()

    def get_layers(self, a_list):
        values = [a_np(a_list)]
        for (a, w, b) in zip(self.activations, self.weights, self.biases):
            before = np.matmul(w, values[-1]) + b
            before = activations[a][0](before)
            values.append(before)
        return values

    def train(self, input_list, target_list):

        layers = self.get_layers(input_list)
        targets = a_np(target_list)

        errors = targets - layers[-1]
        for i in range(len(layers) - 1, 0, -1):
            gradients = activations[self.activations[i - 1]][1](layers[i])
            gradients *= errors
            gradients *= self.learning_rate
            before_layer_t = np.transpose(layers[i-1])
            deltas = gradients * before_layer_t
            self.weights[i - 1] += deltas
            self.biases[i - 1] += gradients

            layer_weight_t = np.transpose(self.weights[i - 1])
            errors = np.matmul(layer_weight_t, errors)

    def save(self, filename):
        save_dict = {
            "layer_sizes": self.layer_sizes,
            "activations": self.activations,
            "learning_rate": self.learning_rate,
            "weights": [],
            "biases": [],
        }

        for w in self.weights:
            save_dict["weights"].append(w.tolist())

        for b in self.biases:
            save_dict["biases"].append(b.tolist())

        with open(filename, "w") as write:
            json.dump(save_dict, write)

    @staticmethod
    def load(filename):
        nn = NeuralNetwork(1, 1)

        with open(filename, "r") as read:
            data = json.load(read)

            nn.layer_sizes = data["layer_sizes"]
            nn.learning_rate = data["learning_rate"]
            nn.activations = data["activations"]

            weights = data["weights"]
            nn.weights.clear()
            for weight in weights:
                nn.weights.append(np.array(weight))

            biases = data["biases"]
            nn.biases.clear()
            for bias in biases:
                nn.biases.append(np.array(bias))
        return nn
