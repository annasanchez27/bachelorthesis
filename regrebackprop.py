import numpy as np
import matplotlib.pyplot as plt

class NeuralNetwork():

    """
    BASIC NEURAL NETWORK WORKING
    """
    def __init__(self,hn):
        self.Ni = 1 #Number of input nodes
        self.Nh = hn #Number of hidden nodes
        self.No = 1 #Number of output nodes
        self.W_layer = [[],[]]
        self.iterations = 1000
        self.error_total_train = []
        self.error_total_test = []

    def initizalize_weights(self):
        """
        Initialization of the weights. Random and Glorot initialization
        :return:
        """
        self.W_layer[0] = np.random.normal(0, 0.3, (self.Nh, self.Ni + 1))
        self.W_layer[1] = np.random.normal(0, 0.3, (self.No, self.Nh + 1))
        #self.W_layer[0] = np.random.normal(0,math.sqrt(2/(self.Ni+self.No)) , (self.Nh, self.Ni + 1))
        #self.W_layer[1] = np.random.normal(0, math.sqrt(2/(self.Ni+self.No)), (self.No, self.Nh + 1))
    def bias_vector(self, X):
        """
        Creation of the bias vector
        :param X: input data
        :return:
        """
        N = np.size(X,1)
        Wo = np.ones([N, 1], dtype = int)
        return Wo

    def feed_forward(self, X, activation_function):
        """
        Feedforward stage
        :param X: input data
        :param activation_function: type of activation function
        :return:
        """
        Y = []
        Wo = self.bias_vector(X)
        if activation_function == "tanh":
            Y.append(np.tanh(np.dot(np.concatenate((X.T, Wo), axis=1), self.W_layer[0].T)))
        if activation_function == "sigmoid":
            Y.append(self.sigmoid(np.dot(np.concatenate((X.T, Wo), axis=1), self.W_layer[0].T)))
        if activation_function == "relu":
            Y.append(self.relu(np.dot(np.concatenate((X.T, Wo), axis=1), self.W_layer[0].T)))
        Y.append(np.dot(np.concatenate((Y[0], Wo), axis=1), self.W_layer[1].T))
        return Y

    def backpropagation(self, input, target, learning_rate, activation_function):
        """
        Backpropagation stage
        :param input: input data
        :param target: target data
        :param learning_rate: learning rate
        :param activation_function: activation function
        :return: vector output (Y[0]: output of hiddden layer, Y[1]: output of the neural network)
        """
        Wo = self.bias_vector(input)
        for i in range(self.iterations):
            Y = self.feed_forward(input, activation_function)
            self.calculate_error(Y,target, 'train')
            derv0 = -1*(target.T - Y[1])
            derv1 = np.concatenate((Y[0],Wo), axis=1)
            dwo = np.dot(derv0.T,derv1)
            c = np.size(self.W_layer[1], 1)
            dwi = np.dot(derv0,np.delete(self.W_layer[1], c-1, axis=1))
            if activation_function == "tanh":
                deriv2 = (1 - Y[0] ** 2)
            if activation_function == "sigmoid":
                deriv2 = Y[0]*(1-Y[0])
            if activation_function == "relu":
                deriv2 = self.reluDerivative(np.dot(np.concatenate((input.T, Wo), axis=1), self.W_layer[0].T))
            delta_h = np.multiply(deriv2,dwi)
            dwi = np.dot(delta_h.T, np.concatenate((input.T,Wo), axis=1))
            self.W_layer[1] = self.W_layer[1] - learning_rate*dwo
            self.W_layer[0] = self.W_layer[0] - learning_rate*dwi
        return Y

    def calculate_error(self, Y,target, t):
        """
        Calculates the error
        :param Y: predicted value
        :param target: target value
        :param t: type (test or train)
        :return:
        """
        error = sum(0.5 * (Y[1] - target.T) ** 2)
        if t == "train":
            self.error_total_train.append(error[0])
        if t == "test":
            self.error_total_test.append(error[0])
        return error[0]
    def plot_train(self, input, target, Y):
        """
        Plot the prediction of the neural network
        :param input: input data
        :param target: target value
        :param Y: predicted value
        :return:
        """
        x_axis = np.array([np.linspace(-1, 1, num=1000)])
        y_axis = np.array(x_axis*x_axis)
        plt.title('Figure representation after training data')
        plt.scatter(input, target, color="m", marker="o", s=20, label="Data points")
        #plt.plot(x_axis[0], y_axis[0], label="")
        plt.scatter(input[0], Y[1], label="Train")
        plt.legend(bbox_to_anchor=(1, 1), loc='upper right', borderaxespad=0.)
        plt.show()
    def plot_test(self, input_train, target_train, input_test, target_test, Y, A):
        """
        Plot test prediction
        :param input_train: input data train
        :param target_train: target data train
        :param input_test: input data test
        :param target_test: target data test
        :param Y: prediction for training data
        :param A: prediction
        :return:
        """
        x_axis = np.array([np.linspace(-1, 1, num=1000)])
        y_axis = np.array(x_axis*x_axis)
        plt.title('Prediction of the function')
        plt.scatter(input_test, target_test, color="m", marker="o", s=20, label="Data points")
        #plt.plot(x_axis[0], y_axis[0], label="Initial function")
        plt.scatter(input_train[0], Y[1], label="Train", marker="o", s=20)
        plt.scatter(input_test[0], A[1], color = 'green', label="Test", marker="+")
        plt.legend(bbox_to_anchor=(1, 1), loc='upper right', borderaxespad=0.)
        plt.show()
    def plot_error(self):
        """
        Plot error in the training stage
        :return:
        """
        plt.ylim(0, 5)
        plt.plot(self.error_total_train, label="Error in each iteration")
        plt.title('Error in the training for normal distribution initialization')
        plt.xlabel('Iteration')
        plt.ylabel('Error')
        plt.show()

    def sigmoid(self,x):
        """
        Sigmoid activation function
        :param x: input
        :return: output
        """
        return 1 / (1 + np.exp(-x))

    def relu(self, X):
        """
        ReLu activation function
        :param X: input
        :return:
        """
        return np.maximum(0, X)

    def reluDerivative(self, x):
        """
        ReLu derivative
        :param x: input
        :return:
        """
        x[x <= 0] = 0
        x[x > 0] = 1
        return x


class CreationData():

    def creation_data(self, numberofdata, type):
        """
        Creation of the training data
        :param numberofdata: number of data points
        :param type: type of example
        :return: input data, target data
        """
        input = np.array([np.linspace(-1, 1, num=numberofdata)])
        target = np.zeros(shape=(1,numberofdata))
        if type == "sinus":
            target = np.array(np.sin(np.pi * input * 2))
        if type == "square":
            target = np.array(input*input)
        if type == "heaviside":
            for i in range(len(input[0])):
                if input[0][i]> 0:
                    target[0][i] = 1
                else:
                    target[0][i] = 0
        if type == "absolute":
            for i in range(len(input[0])):
                if input[0][i]> 0:
                    target[0][i] = input[0][i]
                else:
                    target[0][i] = -input[0][i]

        return input,target

    def trainvstest(self, numberofdata, num_train):
        """
        Division of train/test data
        :param numberofdata: number of points
        :param num_train: number of training points
        :return:
        """
        train = int(numberofdata*num_train/100)
        test = numberofdata - train
        return train,test
if __name__ == "__main__":
    #Choose the percentages of training and test data
    numberofdata = 100
    num_train = 70
    num_test = 30

    learning_rates = [ 0.006,0.005, 0.007,0.008, 0.009]
    learning_rates = sorted(learning_rates)
    activation_functions = ['tanh', 'sigmoid', 'relu']
    hidden_nodes = [2,3,4,5,6,7]
    learning_plot = []
    activation_plot = []
    hidden_plot = []
    error_plot = []
    error = []
    fig = "square"

    CD = CreationData()
    points_train, points_test = CD.trainvstest(numberofdata, num_train)
    input_train, target_train = CD.creation_data(points_train, fig)
    input_test, target_test = CD.creation_data(points_test, fig)
    for act_funct in activation_functions:
        er2 = []
        for lr in learning_rates:
            er = []
            for hn in hidden_nodes:
                NN = NeuralNetwork(hn)
                NN.initizalize_weights()
                Y = NN.backpropagation(input_train,target_train, lr, act_funct)
                #NN.plot_train(input_train,target_train,Y)
                #NN.plot_error()
                A = NN.feed_forward(input_test, act_funct)
                #NN.plot_test(input_train, target_train, input_test, target_test,Y,A)
                er.append(NN.calculate_error(A, target_test, 'test'))
                #learning_plot.append(lr)
                #hidden_plot.append(hn)
                #error_plot.append(NN.calculate_error(A, target_test, 'test'))
                #error.append(NN.calculate_error(A, target_test, 'test'))
            er2.append(er)
        for error, lr in zip(er2,learning_rates):
            plt.plot(hidden_nodes, error)
            plt.scatter(hidden_nodes, error,label = 'Learning rate ' + str(lr), s=40)
        plt.title('Error in the test set for the '+ str(act_funct) + " activation function")
        plt.xlabel('Hidden nodes')
        plt.ylabel('Error')
        plt.legend(bbox_to_anchor=(1, 1), loc='upper right', borderaxespad=0.)
        plt.show()
            #er2.append(er)
