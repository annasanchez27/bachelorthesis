import numpy as np
import matplotlib.pyplot as plt
import math
import seaborn as sn
import pandas as pd

class NeuralNetwork():

    """
    BASIC NEURAL NETWORK WORKING
    """
    def __init__(self, hn):
        self.Ni = 2 #Number of input nodes
        self.Nh = hn #Number of hidden nodes
        self.No = 1 #Number of output nodes
        self.W_layer = [[],[]]
        self.iterations = 10000
        self.error_total_train = []
        self.error_total_test = []
    def sigmoid(self, x):
        """
        Sigmoid activation function
        :param x: input
        :return:
        """
        return 1 / (1 + np.exp(-x))

    def creation_data(self):
        """
        Creation of the training data
        :return:
        """
        inpu = np.array([[0,0,1,1],[0,1,0,1]])
        target = np.array([[0],[1],[1],[0]])
        return inpu,target

    def initizalize_weights(self):
        """
        Initialization of the weights
        :return:
        """
        self.W_layer[0] = np.random.normal(0, 1, (self.Nh, self.Ni + 1))
        self.W_layer[1] = np.random.normal(0, 1, (self.No, self.Nh + 1))

    def bias_vector(self, X):
        """
        Creation of the bias vector
        :param X: input
        :return:
        """
        N = np.size(X,1)
        Wo = np.ones([N, 1], dtype = int)
        return Wo

    def feed_forward(self, X):
        """
        Feedforward step
        :param X: input
        :return:
        """
        Y = []
        Wo = self.bias_vector(X)
        Y.append(self.sigmoid(np.dot(np.concatenate((X.T, Wo), axis=1), self.W_layer[0].T)))
        Y.append(self.sigmoid(np.dot(np.concatenate((Y[0], Wo), axis=1), self.W_layer[1].T)))
        return Y

    def backpropagation(self, input, target,lr):
        """
        Back-propagation algorithm
        :param input: input data
        :param target: target value
        :param lr: learning rate
        :return:
        """
        Wo = self.bias_vector(input)
        for i in range(self.iterations):
            Y = self.feed_forward(input)
            self.calculate_error(Y,target, 'train')
            derv0 = -1*(target - Y[1])
            #NEW
            new = np.multiply(derv0,Y[1]*(1-Y[1]))
            derv1 = np.concatenate((Y[0],Wo), axis=1)
            dwo = np.dot(new.T,derv1)
            c = np.size(self.W_layer[1], 1)
            dwi = np.dot(derv0,np.delete(self.W_layer[1], c-1, axis=1))
            deriv2 = Y[0] * (1 - Y[0])
            delta_h = np.multiply(deriv2,dwi)
            dwi = np.dot(delta_h.T, np.concatenate((input.T,Wo), axis=1))
            self.W_layer[1] = self.W_layer[1] - lr*dwo
            self.W_layer[0] = self.W_layer[0] - lr*dwi

        return Y
    def classification(self, Y,target):
        """
        Classification of the results in TP, FP,TN, FN
        :param Y: predicted values
        :param target: target values
        :return:
        """
        print("Y important")
        print(Y)
        print("Target important")
        print(target)
        axis_x = []
        axis_y = []
        the = np.linspace(0,1,num=100)
        #the = [0.5]
        for t in the:
            result = []
            for x,tar in zip(Y,target):
                if x[0]>t:
                    if tar[0] == 1:
                        result.append('TP')
                    else:
                        result.append('FP')
                else:
                    if tar[0] == 0:
                        result.append('TN')
                    else:
                        result.append('FN')
            tp = result.count('TP')
            fp = result.count('FP')
            tn = result.count('TN')
            fn = result.count('FN')
            labels = ['TP','FP', 'TN', 'FN']

            array = [[tp,fp],[fn,tn]]
            plt.pie([tp,fp,tn,fn], labels=labels, autopct='%1.0f%%')
            plt.legend(bbox_to_anchor=(1, 1), loc='upper right', borderaxespad=0.)
            plt.title("Confusion matrix")
            plt.show()
            df_cm = pd.DataFrame(array, index=[i for i in "01"],
                                 columns=[i for i in "01"])
            plt.figure(figsize=(10, 7))
            sn.heatmap(df_cm, annot=True)
            plt.legend(bbox_to_anchor=(1, 1), loc='upper right', borderaxespad=0.)
            plt.title("Confusion matrix")
            plt.xlabel("Actual value")
            plt.ylabel("Predicted value")
            plt.show()

            sensitivity = tp/(tp+fn)
            specificity = tn/(tn+fp)
            axis_x.append(1-specificity)
            axis_y.append(sensitivity)

        plt.plot(axis_x,axis_y)
        a = np.linspace(0,1,num=100)
        b = a
        plt.plot(a,b,"r--")
        plt.title("ROC Curve")
        plt.xlabel("1-Specificity")
        plt.ylabel("Sensitivity")
        plt.show()

    def calculate_error(self, Y,target, t):
        """
        Calculates the error between the target and the predicted value
        :param Y: predicted value
        :param target: target value
        :param t: type (test or train data)
        :return:
        """
        error = sum(0.5 * (Y[1] - target) ** 2)
        if t == "train":
            self.error_total_train.append(error[0])
        if t == "test":
            print(10*math.log(error[0],10))
            self.error_total_test.append(error[0])
        return error[0]

    def test_data(self):
        """
        Creation of the test data
        :return:
        """
        x1 = np.random.randint(2, size=10000)
        x2 = np.random.randint(2, size=10000)
        y = x1^x2
        target = []
        for i in y:
            target.append([i])
        inpu = np.array([x1, x2])
        return inpu,target

    def plot_error(self):
        """
        Plot the error function per each iteration
        :return:
        """
        plt.plot(self.error_total_train, label="Error in each iteration")
        plt.title('Error in the training for normal distribution initialization')
        plt.xlabel('Iteration')
        plt.ylabel('Error')
        plt.show()


if __name__ == "__main__":
    learning_rates = [ 0.1,0.2,0.3,0.4,0.5]
    learning_rates = sorted(learning_rates)
    hidden_nodes = [2,3,4,5,6,7]

    """
    #To get the results for the xor problem for fixed hyper-parameters
    NN = NeuralNetwork(2)
    NN.initizalize_weights()
    input,target = NN.creation_data()
    Y = NN.backpropagation(input,target,0.3)
    NN.plot_error()
    #NN.classification(Y[1], target,0.5)
    input_test, target_test = NN.test_data()
    A = NN.feed_forward(input_test)
    NN.classification(A[1], target_test)
    """
    #To get the model selection and model evaluation
    er2 = []
    for lr in learning_rates:
        er = []
        for hn in hidden_nodes:
            NN = NeuralNetwork(hn)
            NN.initizalize_weights()
            input, target = NN.creation_data()
            Y = NN.backpropagation(input, target, lr)
            input_test, target_test = NN.test_data()
            A = NN.feed_forward(input_test)
            er.append(NN.calculate_error(A, target_test, 'test'))
        er2.append(er)
    for error, lr in zip(er2, learning_rates):
        plt.plot(hidden_nodes, error)
        plt.scatter(hidden_nodes, error, label='Learning rate ' + str(lr), s=40)
    plt.title('Error in the test set')
    plt.xlabel('Hidden nodes')
    plt.ylabel('Error')
    plt.ylim(0,1)
    plt.legend(bbox_to_anchor=(1, 1), loc='upper right', borderaxespad=0.)
    plt.show()
