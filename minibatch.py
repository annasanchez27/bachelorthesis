import random
import matplotlib.pyplot as plt
import numpy as np


class LinearRegression(object):
    def __init__(self, M):
        self.x = []
        self.y = []
        self.coefficients = []
        self.error_total = []
        for i in range(M+1):
            self.coefficients.append(0)

    def creation_parameters(self, numberofdata):
        """
        Creation of the parameters
        :param numberofdata:
        :return:
        """
        for i in range(0, numberofdata):
            self.x.append(i)
        for x in self.x:
            self.y.append(x + np.random.normal(0, 2, 1))
        self.coefficients = [0,0]
        return self.x, self.y

    def calculate_prediction(self, x):
        """
        Calculate the prediction of a new data point
        :param x: new data point
        :return:
        """
        q = 0
        count = 0
        for coef in self.coefficients:
            q = q + coef*x**count
            count = count + 1
        return q

    def lms(self, rate, n_epoch, size_batch):
        """
        LMS algorithm
        :param rate: learning rate
        :param n_epoch: number of epochs
        :param size_batch: Size of the batch
        :return:
        """
        m = len(self.y)
        for ite in range(n_epoch):
            for i in range(size_batch):
                rand = []
                rand.append(random.randint(0, len(self.x) - 1))
            sum_coef0 = 0
            sum_coef1 = 0
            sum_total = 0
            for a in rand:
                predicted = self.calculate_prediction(self.x[a])
                error = predicted - self.y[a]
                sum_total = sum_total + error**2
                sum_coef0 = sum_coef0 + error
                sum_coef1 = sum_coef1 + error*self.x[a]
            sum_coef0 = 1/m*sum_coef0
            sum_coef1 = 1 / m * sum_coef1
            sum_total = 0.5/m*sum_total
            self.error_total.append(sum_total)
            self.coefficients[0] = self.coefficients[0] - rate*sum_coef0
            self.coefficients[1] = self.coefficients[1] - rate * sum_coef1
        return self.error_total

    def plot(self,x,y,coef, error_total):
        """
         This function plots the original function, the training data and the expected regression
         :param x: training data
         :param y: training data
         :param coef: coefficients calculated of the regression
         :return:
         """
        fx= []
        rx = np.linspace(0, 10, num=10000)

        # printting the actual points (with Noise)
        plt.scatter(self.x, self.y, color="m", marker="o", s=20, label="Data points of signal + noise")

        # ploting the sinus function (with NO Noise)
        """
        a = np.arange(0, 2 * np.pi, 0.1)
        b = np.sin(a)
        plt.plot(a, b, label="Function sinus")
        """


        for i in range(len(rx)):
            count = 0
            for a in coef:
                if count == 0:
                    fx.append(a * rx[i] ** count)
                else:
                    fx[i] = fx[i] + a * rx[i] ** count
                count = count + 1

        plt.plot(rx, fx, label="Predicted regression")
        plt.legend(bbox_to_anchor=(1, 1), loc='upper right', borderaxespad=0.)
        plt.title('Mini-Batch Gradient Descent prediction')
        plt.xlabel('x')
        plt.ylabel('y')
        # function to show plot
        plt.show()
        plt.plot(error_total)
        # plt.legend(bbox_to_anchor=(1, 1), loc='upper right', borderaxespad=0.)
        plt.title("Cost function in different iterations for Mini-Batch Gradient Descent")
        plt.xlabel('Iterations')
        plt.ylabel('Cost function')
        plt.show()


if __name__ == "__main__":
    error_lr = []
    rate = 0.001
    print("Welcome to the Mini batch gradient program. Please choose the size of the batch")
    size_batch = input()
    LR = LinearRegression(1)
    LR.creation_parameters(10)
    error = LR.lms(0.1, 100, int(size_batch))
    LR.plot(LR.x,LR.y, LR.coefficients, error)