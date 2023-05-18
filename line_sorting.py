import numpy as np

def network():

    class NeuronLayer():
        def __init__(self, number_of_neurons, number_of_inputs_per_neuron):
            self.synaptic_weights = 2 * np.random.random((number_of_inputs_per_neuron, number_of_neurons)) - 1

    class NeuralNetwork():
        def __init__(self, layer1, layer2, layer3):
            self.layer1 = layer1
            self.layer2 = layer2
            self.layer3 = layer3

        def sigmoid(self, x):
            return 1 / (1 + np.exp(-x))

        def sigmoid_derivative(self, x):
            return x * (1 - x)

        def train(self, training_set_inputs, training_set_outputs, number_of_training_iterations):
            for i in range(number_of_training_iterations):

                output_from_layer_1, output_from_layer_2, output_from_layer_3 = self.think(training_set_inputs)

                layer3_error = training_set_outputs - output_from_layer_3
                layer3_delta = layer3_error * self.sigmoid_derivative(output_from_layer_3)

                layer2_error = layer3_delta.dot(self.layer3.synaptic_weights.T)
                layer2_delta = layer2_error * self.sigmoid_derivative(output_from_layer_2)

                layer1_error = layer2_delta.dot(self.layer2.synaptic_weights.T)
                layer1_delta = layer1_error * self.sigmoid_derivative(output_from_layer_1)

                layer1_adjustment = training_set_inputs.T.dot(layer1_delta)
                layer2_adjustment = output_from_layer_1.T.dot(layer2_delta)
                layer3_adjustment = output_from_layer_2.T.dot(layer3_delta)

                self.layer1.synaptic_weights += layer1_adjustment
                self.layer2.synaptic_weights += layer2_adjustment
                self.layer3.synaptic_weights += layer3_adjustment

        def think(self, inputs):
            output_from_layer1 = self.sigmoid(np.dot(inputs, self.layer1.synaptic_weights))
            output_from_layer2 = self.sigmoid(np.dot(output_from_layer1, self.layer2.synaptic_weights))
            output_from_layer3 = self.sigmoid(np.dot(output_from_layer2, self.layer3.synaptic_weights))
            return output_from_layer1, output_from_layer2, output_from_layer3

    np.random.seed(1)

    # number of neurons, number of inputs
    layer1 = NeuronLayer(7, 9)
    layer2 = NeuronLayer(7, 7)
    layer3 = NeuronLayer(3, 7)

    neural_network = NeuralNetwork(layer1, layer2, layer3)

    training_set_inputs = np.array([[1, 1, 1, 0, 0, 0, 0, 0, 0], 
                                    [0, 1, 0, 0, 1, 0, 0, 1, 0], 
                                    [0, 0, 0, 1, 1, 1, 0, 0, 0], 
                                    [0, 0, 1, 0, 0, 1, 0, 0, 1], 
                                    [1, 0, 0, 1, 0, 0, 1, 0, 0], 
                                    [0, 0, 0, 0, 0, 0, 1, 1, 1], 
                                    [1, 0, 0, 0, 1, 0, 0, 0, 1], 
                                    [0, 0, 1, 0, 1, 0, 1, 0, 0],])

    training_set_outputs = np.array([[1, 0, 0], 
                                     [0, 1, 0], 
                                     [1, 0, 0], 
                                     [0, 1, 0], 
                                     [0, 1, 0], 
                                     [1, 0, 0], 
                                     [0, 0, 1], 
                                     [0, 0, 1],])

    neural_network.train(training_set_inputs, training_set_outputs, 10000)

    hidden_output_1, hidden_output_2, output = neural_network.think(np.array([0.31, 0.21, 0.11, 0.96, 0.85, 0.91, 0.22, 0.29, 0.14]))

    # Horizontal: [0.31, 0.21, 0.11, 0.96, 0.85, 0.91, 0.22, 0.29, 0.14]
    # Verticle: [0.85, 0.21, 0.11, 0.96, 0.31, 0.17, 0.91, 0.22, 0.13]
    # Diagonal: [0.95, 0.17, 0.11, 0.2, 0.92, 0.31, 0.09, 0.22, 0.87]

    print(output)

    print("\nHorizontal output:", float(output[0]))
    print("Vertical output:", float(output[1]))
    print("Diagonal output:", float(output[2]), "\n")

network()
