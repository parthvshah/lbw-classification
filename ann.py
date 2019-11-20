"""
A module to implement the stochastic gradient descent learning
algorithm for a feedforward neural network.  Gradients are calculated
using backpropagation.  Note that I have focused on making the code
simple, easily readable, and easily modifiable.  It is not optimized,
and omits many desirable features.
"""

import random
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

def sorted_func(x):
    return x[1]

def sigmoid(z):
    return 1.0/(1.0+np.exp(-z))

def sigmoid_prime(z):
    return sigmoid(z)*(1-sigmoid(z))

def relu(z):
    if(z.all()<0):
        return 0
    else:
        return z

def relu_prime(z):
    if(z.all()<0):
        return 0
    else:
        return 1

def tanh(z):
    return np.tanh(z)

def tanh_prime(z):
    return 1.0 - np.tanh(z)**2

def act(z, choice):
    if(choice=="sigmoid"):
        return sigmoid(z)
    elif(choice=="relu"):
        return relu(z)
    elif(choice=="tanh"):
        return tanh(z)

def act_prime(z, choice):
    if(choice=="sigmoid"):
        return sigmoid_prime(z)
    elif(choice=="relu"):
        return relu_prime(z)
    elif(choice=="tanh"):
        return tanh_prime(z)
    

def normalize(df):
    norm_df = (df-df.min())/(df.max()-df.min())
    return norm_df

def read_data(class1, class2):
    data_class1 = pd.read_csv(class1)
    data_class2 = pd.read_csv(class2)
    data = data_class1.append(data_class2)
    return data

def create_vectors(data, result):
    x = data.drop(result, axis=1)
    y = data[result]
    return x, y

def split(x, y, test_percentage):
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = test_percentage)
    return x_train, x_test, y_train, y_test

def create_input(x, y):
    output = []
    x = x.to_numpy()
    y = y.to_numpy()
    for i in range(len(x)):
        output.append((x[i], y[i]))
    return np.array(output)


class Network(object):
    def __init__(self, sizes):
        """The list ``sizes`` contains the number of neurons in the
        respective layers of the network.  For example, if the list
        was [2, 3, 1] then it would be a three-layer network, with the
        first layer containing 2 neurons, the second layer 3 neurons,
        and the third layer 1 neuron.  The biases and weights for the
        network are initialized randomly, using a Gaussian
        distribution with mean 0, and variance 1.  Note that the first
        layer is assumed to be an input layer, and by convention we
        won't set any biases for those neurons, since biases are only
        ever used in computing the outputs from later layers."""
        self.num_layers = len(sizes)
        self.sizes = sizes
        self.biases = [np.random.randn(y, 1) for y in sizes[1:]]
        self.weights = [np.random.randn(y, x)
                        for x, y in zip(sizes[:-1], sizes[1:])]

    def feedforward(self, a):
        """Return the output of the network if ``a`` is input."""
        for b, w in zip(self.biases, self.weights):
            a = act(np.dot(w, a)+b, "tanh")
        return a

    def train(self, training_data, epochs, mini_batch_size, eta,
            test_data=None, test_data_bool=False):
        """Train the neural network using mini-batch stochastic
        gradient descent.  The ``training_data`` is a list of tuples
        ``(x, y)`` representing the training inputs and the desired
        outputs.  The other non-optional parameters are
        self-explanatory.  If ``test_data`` is provided then the
        network will be evaluated against the test data after each
        epoch, and partial progress printed out.  This is useful for
        tracking progress, but slows things down substantially."""
        if test_data_bool:
            n_test = len(test_data)
        n = len(training_data)
        for j in range(epochs):
            random.shuffle(training_data)
            mini_batches = [
                training_data[k:k+mini_batch_size]
                for k in range(0, n, mini_batch_size)]
            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch, eta)
            if test_data_bool:
                print("Epoch {0}: {1} / {2}".format(
                    j, self.evaluate(test_data), n_test))
            else:
                print("Epoch {0} complete".format(j))

    def update_mini_batch(self, mini_batch, eta):
        """Update the network's weights and biases by applying
        gradient descent using backpropagation to a single mini batch.
        The ``mini_batch`` is a list of tuples ``(x, y)``, and ``eta``
        is the learning rate."""
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        for x, y in mini_batch:
            delta_nabla_b, delta_nabla_w = self.backprop(x, y)
            nabla_b = [nb+dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw+dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
        self.weights = [w-(eta/len(mini_batch))*nw
                        for w, nw in zip(self.weights, nabla_w)]
        self.biases = [b-(eta/len(mini_batch))*nb
                       for b, nb in zip(self.biases, nabla_b)]

    def backprop(self, x, y):
        """Return a tuple ``(nabla_b, nabla_w)`` representing the
        gradient for the cost function C_x.  ``nabla_b`` and
        ``nabla_w`` are layer-by-layer lists of numpy arrays, similar
        to ``self.biases`` and ``self.weights``."""
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        # feedforward
        activation = x
        activations = [x] # list to store all the activations, layer by layer
        zs = [] # list to store all the z vectors, layer by layer
        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, activation)+b
            zs.append(z)
            activation = act(z, "tanh")
            activations.append(activation)
        # backward pass
        delta = self.cost_derivative(activations[-1], y) * \
            act_prime(zs[-1], "tanh")
        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta, activations[-2].transpose())
        for l in range(2, self.num_layers):
            z = zs[-l]
            sp = act_prime(z, "tanh")
            delta = np.dot(self.weights[-l+1].transpose(), delta) * sp
            nabla_b[-l] = delta
            nabla_w[-l] = np.dot(delta, activations[-l-1].transpose())
        return (nabla_b, nabla_w)

    def evaluate(self, test_data):
        """Return the number of test inputs for which the neural
        network outputs the correct result. Note that the neural
        network's output is assumed to be the index of whichever
        neuron in the final layer has the highest activation."""
        # test_results = [(np.argmax(self.feedforward(x)), y)
        #                 for (x, y) in test_data]
        test_results = [(self.feedforward(x), y)
                        for (x, y) in test_data]
        print_results = sorted(test_results, key=sorted_func)
        for result in print_results:
            print(tanh(sum(result[0][0])), result[1])
        return sum(int(x == y) for (x, y) in test_results)

    def cost_derivative(self, output_activations, y):
        """Return the vector of partial derivatives \partial C_x /
        \partial a for the output activations."""
        return (output_activations-y)


if __name__ == "__main__":
    data = read_data("./data/lbw0.csv", "./data/lbw1.csv")
    data = normalize(data)
    x, y = create_vectors(data, "reslt")
    x_train, x_test, y_train, y_test = split(x, y, 0.20)

    # print(x_train, x_test, y_train, y_test, sep="\n")

    train_input = create_input(x_train, y_train) 
    test_input = create_input(x_test, y_test)

    network = Network([8,8,1])
    network.train(train_input, 1000, 5, 0.05)

    result = network.evaluate(test_input)
    print(result, len(test_input), result/len(test_input))
