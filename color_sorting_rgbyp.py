import numpy as np
import pygame as py

def network():
    global red_list, green_list, blue_list, yellow_list, purple_list

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

    layer1 = NeuronLayer(9, 3)
    layer2 = NeuronLayer(9, 9)
    layer3 = NeuronLayer(5, 9)

    neural_network = NeuralNetwork(layer1, layer2, layer3)

    training_set_inputs = np.array([[1.00, 0.00, 0.00], 
                                    [0.94, 0.02, 0.03], 
                                    [0.99, 0.22, 0.24], 

                                    [0.00, 1.00, 0.00], 
                                    [0.05, 0.93, 0.04], 
                                    [0.30, 0.96, 0.28], 

                                    [0.00, 0.00, 1.00], 
                                    [0.07, 0.08, 0.94],
                                    [0.19, 0.21, 0.98],
                                    
                                    [1.00, 1.00, 0.00],
                                    [0.94, 0.93, 0.14],
                                    [0.92, 0.87, 0.13],
                                    
                                    [1.00, 0.00, 1.00],
                                    [0.93, 0.12, 0.91],
                                    [0.73, 0.18, 0.68],])

    training_set_outputs = np.array([[1, 0, 0, 0, 0], 
                                     [1, 0, 0, 0, 0], 
                                     [1, 0, 0, 0, 0], 
                                     [0, 1, 0, 0, 0], 
                                     [0, 1, 0, 0, 0], 
                                     [0, 1, 0, 0, 0], 
                                     [0, 0, 1, 0, 0], 
                                     [0, 0, 1, 0, 0],
                                     [0, 0, 1, 0, 0],
                                     [0, 0, 0, 1, 0],
                                     [0, 0, 0, 1, 0],
                                     [0, 0, 0, 1, 0],
                                     [0, 0, 0, 0, 1],
                                     [0, 0, 0, 0, 1],
                                     [0, 0, 0, 0, 1],])

    neural_network.train(training_set_inputs, training_set_outputs, 10000)

    r, g, b = input("Enter a color in RGB form: ").split(", ")

    color_list = [[255, 0, 0],
                  [0, 255, 0],
                  [0, 0, 255],
                  [255, 255, 0],
                  [255, 0, 255],

                  [7, 132, 12], 
                  [32, 41, 203], 
                  [149, 45, 42],
                  [84, 225, 115],
                  [135, 196, 129],
                  [114, 50, 179],
                  [36, 231, 169],
                  [153, 71, 68],
                  [205, 109, 79],
                  [93, 54, 229],
                  [29, 112, 46],
                  [189, 58, 68],
                  [217, 71, 68],
                  [26, 71, 210],
                  [178, 44, 182],
                  [148, 32, 154],
                  [221, 203, 9],
                  [42, 39, 172],
                  [54, 60, 210],
                  [194, 46, 186],
                  [152, 24, 143],
                  [194, 199, 57],
                  [225, 248, 102],
                  [147, 157, 54],
                  [199, 209, 130],
                  
                  [int(r), int(g), int(b)],]

    red_list = []
    green_list = []
    blue_list = []
    yellow_list = []
    purple_list = []

    for i in color_list:

        hidden_output_1, hidden_output_2, output = neural_network.think(np.array([i[0], i[1], i[2]]))

        if list(output).index(max(output)) == 0:
            red_list.append(i)
        
        if list(output).index(max(output)) == 1:
            green_list.append(i)
        
        if list(output).index(max(output)) == 2:
            blue_list.append(i)
        
        if list(output).index(max(output)) == 3:
            yellow_list.append(i)
        
        if list(output).index(max(output)) == 4:
            purple_list.append(i)

    print("Reds:", red_list)
    print("Greens:", green_list)
    print("Blues:", blue_list)
    print("Yellows:", yellow_list)
    print("Purples:", purple_list)

network()

screen_width = 800
screen_height = 480
screen = py.display.set_mode((screen_width, screen_height))
py.display.set_caption("Colors")

width = 40
height = 40

red_x = 40
red_y = 40

green_x = 40
green_y = 120

blue_x = 40
blue_y = 200

yellow_x = 40
yellow_y = 280

purple_x = 40
purple_y = 360

screen.fill((50, 50, 50))

for i in red_list:
    py.draw.rect(screen, i, ((red_x, red_y), (width, height)))
    red_x += 80

for i in green_list:
    py.draw.rect(screen, i, ((green_x, green_y), (width, height)))
    green_x += 80
    
for i in blue_list:
    py.draw.rect(screen, i, ((blue_x, blue_y), (width, height)))
    blue_x += 80

for i in yellow_list:
    py.draw.rect(screen, i, ((yellow_x, yellow_y), (width, height)))
    yellow_x += 80

for i in purple_list:
    py.draw.rect(screen, i, ((purple_x, purple_y), (width, height)))
    purple_x += 80

window_open = True
while window_open:
    for event in py.event.get():
        if event.type == py.QUIT:
            window_open = False

    py.display.update()

py.quit()
