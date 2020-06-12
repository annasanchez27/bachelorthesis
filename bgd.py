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
        :param numberofdata: number of data points
        :return:
        """
        for i in range(0, numberofdata):
            self.x.append(i)
            self.y.append(i)
        self.x = [1, 2, 4, 3, 5, 6, 7, 8, 9, 10]
        self.y = [1, 3, 3, 2, 5, 6, 8, 8, 10, 10]
        self.coefficients = [0,0]
        return self.x, self.y

    def calculate_prediction(self, x):
        """
        Predicts the value of a new data point
        :param x: new data point
        :return:
        """
        q = 0
        count = 0
        for coef in self.coefficients:
            q = q + coef*x**count
            count = count + 1
        return q

    def lms(self, rate, n_epoch):
        """
        lms algorithm
        :param rate: learning rate
        :param n_epoch: number of epochs
        :return:
        """
        m = len(self.y)
        for ite in range(n_epoch):
            sum_coef0 = 0
            sum_coef1 = 0
            sum_total = 0
            for x, y in zip(self.x, self.y):
                predicted = self.calculate_prediction(x)
                error = predicted - y
                sum_total = sum_total + error**2
                sum_coef0 = sum_coef0 + error
                sum_coef1 = sum_coef1 + error*x
            sum_coef0 = 1/m*sum_coef0
            sum_coef1 = 1 / m * sum_coef1
            sum_total = 0.5/m*sum_total
            self.error_total.append(sum_total)
            self.coefficients[0] = self.coefficients[0] - rate*sum_coef0
            self.coefficients[1] = self.coefficients[1] - rate * sum_coef1
        return self.error_total


    def plot(self,coef, error_total, rate):
        """
         This function plots the original function, the training data and the expected regression
         :param x: training data
         :param y: training data
         :param coef: coefficients calculated of the regression
         :return:
         """
        fx= []
        rx = np.linspace(0, 10, num=1000)

        # printting the actual points (with Noise)
        plt.scatter(self.x, self.y, color="m", marker="o", s=20, label="Data points")
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
        plt.title('Batch Gradient Descent prediction')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.ylim(0, 10)
        # function to show plot
        plt.show()
        plt.plot(error_total, label="Cost function for learning rate =" + str(rate))
        plt.legend(bbox_to_anchor=(1, 1), loc='upper right', borderaxespad=0.)
        plt.title("Cost function in different iterations for Batch Gradient Descent")
        plt.xlabel('Iterations')
        plt.ylabel('Cost function')
        plt.show()


if __name__ == "__main__":
    error_lr = []
    LR = LinearRegression(1)
    LR.creation_parameters(10)
    rate = 0.01
    error = LR.lms(rate, 50)
    LR.plot(LR.coefficients, error, rate)