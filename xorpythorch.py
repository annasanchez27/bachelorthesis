import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.autograd import Variable

class XOR(nn.Module):
    def __init__(self,hidden_dim):
        super(XOR, self).__init__()
        input_dim = 2
        output_dim = 1
        self.first_layer = nn.Linear(input_dim, hidden_dim)
        self.second_layer = nn.Linear(hidden_dim, output_dim)
        self.error_total_train = []
        self.error_total_test = []

    def forward(self, x):
        """
        Forward step in the neural network
        :param x: input
        :return:
        """
        x = self.first_layer(x)
        x = torch.tanh(x)
        x = self.second_layer(x)
        return x

    def weights_init(self, model):
        """
        Weight initialization
        :param model: model
        :return:
        """
        for m in model.modules():
            if type(m) == nn.Linear:
                m.weight.data.normal_(0, 1)
    def SGD(self, X,Y, lr):
        """
        Stochastic Gradient Descent algorithm
        :param X: data points
        :param Y: target value
        :param lr: learning rate
        :return:
        """
        loss_func = nn.MSELoss()
        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)
        for epoch in range(1000):
            for j in range(X.size(0)):
                data_point = np.random.randint(X.size(0))
                x_var = Variable(X[data_point], requires_grad=False)
                y_var = Variable(Y[data_point], requires_grad=False)
                optimizer.zero_grad()
                y_pred = model(x_var)
                loss = loss_func.forward(y_pred, y_var)
                loss.backward()
                optimizer.step()
            if epoch % 100 == 0:
                print("Epoch {: >8} Loss: {}".format(epoch, loss.data.numpy()))

    def calculate_accuracy(self, Y, target, type):
        """
        Calculates accuracy
        :param Y: predicted value
        :param target: target value
        :param type: type (test or train)
        :return:
        """
        the = [0.5]
        for t in the:
            result = []
            for x, tar in zip(Y, target):
                if x[0] > t:
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
        if type == "train":
            self.error_total_train.append((tp + tn) / (tp + tn + fp + fn))
        if type == "test":
            self.error_total_test.append((tp + tn) / (tp + tn + fp + fn))


class Classification():
    def plot(self,X,model):
        model_params = list(model.parameters())
        model_weights = model_params[0].data.numpy()
        print(model_weights)
        model_bias = model_params[1].data.numpy()

        plt.scatter(X.numpy()[[0, -1], 0], X.numpy()[[0, -1], 1], s=50)
        plt.scatter(X.numpy()[[1, 2], 0], X.numpy()[[1, 2], 1], c='red', s=50)

        x_1 = np.arange(-0.1, 1.1, 0.1)
        y_1 = ((x_1 * model_weights[0, 0]) + model_bias[0]) / (-model_weights[0, 1])
        plt.plot(x_1, y_1)

        x_2 = np.arange(-0.1, 1.1, 0.1)
        y_2 = ((x_2 * model_weights[1, 0]) + model_bias[1]) / (-model_weights[1, 1])
        plt.plot(x_2, y_2)
        plt.legend(["neuron_1", "neuron_2"], loc=8)
        plt.show()

    def test_data(self):
        """
        Creates test data
        :return:
        """
        x1 = np.random.randint(2, size=10000)
        x2 = np.random.randint(2, size=10000)
        y = x1 ^ x2
        target = []
        for i in y:
            target.append([i])
        inpu = np.array([x1, x2])
        return inpu.T, target
    def train_data(self):
        """
        Creates the train data
        :return:
        """
        X = torch.Tensor([[0, 0], [0, 1], [1, 0], [1, 1]])
        Y = torch.Tensor([0, 1, 1, 0]).view(-1, 1)
        return X,Y

    def classification(self,Y,target):
        """
        Classification depending if it is TP,FP,TN,FN
        :param Y: predicted value
        :param target: target value
        :return:
        """
        axis_x = []
        axis_y = []
        the = np.linspace(0,1,num=100)
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
    def predict_test(self, inpu,targ):
        """
        Prediction of the test set
        :param inpu: input
        :param targ: target value
        :return:
        """
        torch_inpu = torch.FloatTensor(inpu)
        torch_targ = torch.FloatTensor(targ)
        list = []
        for input, target in zip(torch_inpu, torch_targ):
            output = model(input)
            list.append(output.tolist())
        return list


if __name__ == "__main__":

    hidden_nodes = [1, 2, 3,4,5]
    learning_rates = [0.01, 0.05,0.1,0.7,0.9]
    error_train = []
    error_test = []

    clasi = Classification()
    X, Y = clasi.train_data()
    inpu, targ = clasi.test_data()

    for lr in learning_rates:
        error_test2 = []
        for hn in hidden_nodes:
            model = XOR(hn)
            model.weights_init(model)
            model.SGD(X, Y, lr)
            predicted = clasi.predict_test(inpu, targ)
            model.calculate_accuracy(predicted, targ, "test")
            error_test2.append(model.error_total_test)
        error_test.append(error_test2)
    for error, lr in zip(error_test, learning_rates):
        plt.plot(hidden_nodes, error)
        plt.scatter(hidden_nodes, error, label='Learning rate ' + str(lr), s=40)
    plt.title("Accuracy in the test set")
    plt.xlabel('Hidden nodes')
    plt.ylabel('Accuracy')
    plt.legend(bbox_to_anchor=(1, 1), loc='upper right', borderaxespad=0.)
    plt.show()
