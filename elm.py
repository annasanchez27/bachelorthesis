import numpy as np
import matplotlib.pyplot as plt
import math

class Network(object):

    def __init__(self, input_dim, hidden_dim, output_dim):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.weights = []
        self.bias = []
        self.x = []
        self.y = []

    def creation_parameters_example(self, numberofdata, sigma, type):
        """
        Creates the random parameters (biases, weights, x and target values).
        :param numberofdata:
        :param type:
        :return:
        """

        # self.weights = np.random.randn(self.hidden_dim)*10
        # self.bias = np.random.randn(self.hidden_dim)
        self.weights = np.random.normal(0, sigma, self.hidden_dim)
        self.bias = np.random.normal(0, sigma, self.hidden_dim)
        self.x = np.linspace(-1, 1, num=numberofdata)

        if type == "square":
            for i in range(len(self.x)):
                self.y.append(self.x[i] * self.x[i])
        if type == "sinus":
            self.y = np.sin(np.pi * self.x * 2)
        if type == "absolute":
            for i in range(len(self.x)):
                if self.x[i] < 0:
                    self.y[i] = -self.x[i]
                else:
                    self.y[i] = self.x[i]
        if type == "heaviside":
            for i in range(len(self.x)):
                if self.x[i] > 0:
                    self.y[i] = 1
                else:
                    self.y[i] = 0

    def plot(self, hidden_functions, output_function):
        """
        Function that plots the original function, data points, output of hidden perceptons and final output
        :param hidden_functions:
        :param output_function:
        :return:
        """
        plt.ylim(0, 1)
        count = 0

        plt.plot(self.x, self.y, label="Initial function")
        plt.scatter(self.x, self.y, color="m", marker="o", s=20, label="Data points")

        for y in hidden_functions:
            plt.plot(self.x, y, label="Output hidden neuron #" + str(count), linestyle="--")
            count = count + 1

        plt.plot(self.x, output_function, label="Predicted function")
        plt.legend(loc='upper right', borderaxespad=0.)
        plt.xlabel('x')
        plt.ylabel('t')
        plt.title("ELM algorithm results")

        plt.show()

    def tanh(self, x):
        """
        Activation function
        :param x: Input
        :return:
        """
        return 2 / (1 + np.exp(-2 * x)) - 1

    def hidden_layer_function(self):
        """
        Calculates the functions of the output of the hidden perceptons
        :return:
        """
        y = []
        hidden_layer = []
        for i in range(len(self.weights)):
            for x in self.x:
                retu = self.tanh(self.weights[i] * x + self.bias[i])
                y.append(retu)
            hidden_layer.append(y)
            y = []
        return hidden_layer

    def final_layer(self, x):
        """
        Calculates the final output of the neural network
        :param x:
        :return:
        """
        coef = self.coef_calculation()
        h = self.H_creation(x)
        output = np.dot(h, coef)
        return output

    def H_creation(self, x):
        """
        Creates the matrix that will be used for linear regression
        :param x:
        :return:
        """
        h_matrix = []
        row = []
        new_row = []
        new_h_matrix = []

        for a in x:
            for w, b in zip(self.weights, self.bias):
                row.append(w * a + b)
            h_matrix.append(row)
            row = []

        for row in h_matrix:
            for element in row:
                new_element = self.tanh(element)
                new_row.append(new_element)
            new_h_matrix.append(new_row)
            new_row = []
        return new_h_matrix

    def coef_calculation(self):
        """
        Calculates the coeficients of the last layer with linear regression
        :return:
        """
        h_matrix = self.H_creation(self.x)
        imatrix = np.linalg.pinv(h_matrix)
        return np.dot(imatrix, self.y)

    def calculate_error(self, y, w):
        """
        Calculates the error
        :param y: target value
        :param w: predicted value
        :return:
        """
        n = len(y)
        error = 0
        for i in range(len(y)):
            a = (y[i] - w[i])
            error = error + pow(a, 2)
        return error

    def plot_error(self, numberofdata):
        """
        Plots error function
        :param numberofdata: number of data points
        :return:
        """
        erms = []
        erms_sigma = []
        erms_layer = []
        x_axis = []
        y_axis = []
        plt.ylim(0, 2)
        axis_layer = np.linspace(1, 10, num=10)
        axis_sigma = np.around(np.linspace(0, 5, num=10), decimals=2)

        for i in range(10):
            print(i)
            self.__init__(1, i + 1, 1)
            self.creation_parameters_example(numberofdata, 1, "sinus")
            error = self.calculate_error(self.y, self.final_layer(self.x))
            error = math.sqrt(error / len(self.x))
            erms.append(error)

        for i in axis_sigma:
            self.__init__(1, 3, 1)
            self.creation_parameters_example(numberofdata, i, "sinus")
            error = self.calculate_error(self.y, self.final_layer(self.x))
            error = math.sqrt(error / len(self.x))
            erms_sigma.append(error)

        plt.plot(axis_layer, erms, 'o-', label="Error for the training data")
        plt.legend(bbox_to_anchor=(1, 1), loc='upper right', borderaxespad=0.)
        plt.title('Error function for different hidden perceptons')
        plt.xlabel('Hidden perceptons')
        plt.ylabel('Erms')
        plt.show()

        plt.plot(axis_sigma, erms_sigma, 'o-', label="Error for the training data")
        plt.legend(bbox_to_anchor=(1, 1), loc='upper right', borderaxespad=0.)
        plt.title('Error function for different values of sigma')
        plt.xlabel('Value of sigma')
        plt.ylabel('Erms')
        plt.show()
        for i, a in zip(range(10), axis_layer):
            for j in axis_sigma:
                self.__init__(1, i, 1)
                self.creation_parameters_example(numberofdata, j, "square")
                error = self.calculate_error(self.y, self.final_layer(self.x))
                error = math.sqrt(error / len(self.x))
                erms_layer.append(error)
                x_axis.append(a)
                y_axis.append(j)
        erms_layer = np.array(erms_layer)
        fig = plt.figure()
        ax = fig.gca(projection='3d')
        ax.plot_trisurf(x_axis, y_axis, erms_layer)

        plt.show()


if __name__ == "__main__":

    num_train = 70
    num_test = 30
    points = 50
    NN = Network(1, 3, 1)
    figures = ["square", "sinus", "absolute", "heaviside"]

    for figure in figures:
        NN.creation_parameters_example(points, 0.3, figure)
        hidden_functions = NN.hidden_layer_function()
        final_function = NN.final_layer(NN.x)
        NN.plot(hidden_functions, final_function)

    NN.plot_error(points)