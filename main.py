import matplotlib.pyplot
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

import numpy
import imageio
import pandas as pd
from NeuralNetwor import NeuralNetwork

input_nodes = 784
hidden_nodes = 200
output_nodes = 10
learning_rate = 0.1
train = False
neural_network = NeuralNetwork(input_nodes, hidden_nodes, output_nodes, learning_rate)
try:
    df = pd.read_csv("weight_input_hidden.csv")
except ValueError:
    train = True

if train:
    training_data_file = open("datasets/mnist_train.csv", 'r')
    training_data_list = training_data_file.readlines()
    training_data_file.close()

    epochs = 10

    for e in range(epochs):
        for record in training_data_list:
            all_values = record.split(',')
            inputs = (numpy.asfarray(all_values[1:]) / 255.0 * 0.99) + 0.01
            targets = numpy.zeros(output_nodes) + 0.01
            targets[int(all_values[0])] = 0.99
            neural_network.train(inputs, targets)
            pass
        pass

    print("real weight_input_hidden {0}".format(neural_network.weight_input_hidden))
    print("real weight_input_hidden size {0} size1 {1}".format(len(neural_network.weight_input_hidden),
                                                               len(neural_network.weight_input_hidden[0])))
    print("real weight_hidden_output {0}".format(neural_network.weight_hidden_output))
    print("real weight_hidden_output size {0} size1 {1}".format(len(neural_network.weight_hidden_output),
                                                                len(neural_network.weight_hidden_output[0])))

    numpy.savetxt('weight_input_hidden.csv', neural_network.weight_input_hidden, delimiter=",")
    numpy.savetxt('weight_hidden_output.csv', neural_network.weight_hidden_output, delimiter=",")
else:
    training_data_file = open("weight_hidden_output.csv", 'r')
    training_data_list = [map(float, line) for line in training_data_file.readlines()]
    training_data_file.close()
    neural_network.weight_hidden_output = training_data_list
    print("weight_hidden_output")
    print(training_data_list)

    training_data_file1 = open("weight_input_hidden.csv", 'r')
    training_data_list1 = [map(float, line) for line in training_data_file1.readlines()]
    training_data_file1.close()
    neural_network.weight_input_hidden = training_data_list1
    print("weight_input_hidden")
    print(training_data_list)

img_array = imageio.imread('numbers_images/own_3.png', as_gray=True)

img_data = 255.0 - img_array.reshape(784)

img_data = (img_data / 255.0 * 0.99) + 0.01
print("min = ", numpy.min(img_data))
print("max = ", numpy.max(img_data))

outputs = neural_network.query(img_data)
print(outputs)

label = numpy.argmax(outputs)
print("network says ", label)

img = mpimg.imread('numbers_images/{0}.png'.format(label))
imgplot = plt.imshow(img)
plt.show()

# matplotlib.pyplot.imshow(img_data.reshape(28, 28), cmap='Greys', interpolation='None')
