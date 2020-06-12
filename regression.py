import random
import matplotlib.pyplot as plt
import numpy as np
import math

def plot(x,y,coef):
    """
     This function plots the original function, the training data and the expected regression
     :param x: training data
     :param y: training data
     :param coef: coefficients calculated of the regression
     :return:
     """
    plt.xlim(0,6)
    plt.ylim(-1.5,1.5)
    rx = np.linspace(0, 6, num=1000)
    fx = []

    # printting the actual points (with Noise)
    plt.scatter(x, y, color="m", marker="o", s=20, label="Data points")

    # ploting the sinus function (with NO Noise)
    a = np.arange(0, 2 * np.pi, 0.1)
    b = np.sin(a)
    plt.plot(a, b, label="Initial function")

    # plotting the prediction made with the regression

    for i in range(len(rx)):
        count = 0
        for a in coef:
            if count == 0:
                fx.append(a * rx[i] ** count)
            else:
                fx[i] = fx[i] + a * rx[i] ** count
            count = count + 1

    plt.plot(rx, fx, label="Predicted function")

    # putting labels
    plt.legend(bbox_to_anchor=(1, 1), loc='upper right', borderaxespad=0.)
    plt.title('Regression for M = '+ str(len(coef)-1))
    plt.xlabel('x')
    plt.ylabel('t')
    # function to show plot
    plt.show()


def mle_matrix_calculation_moorse(degree, x, y):
    """

    :param degree: degree of the fitting polynomial
    :param x:
    :param y:
    :return: coefficients
    """
    matrix = []
    row = []
    for a in x:
        for n in range(degree+1):
            row.append(pow(a,n))
        matrix.append(row)
        row = []
    fmatrix = np.linalg.pinv(matrix)
    coeficients = np.dot(fmatrix, y)
    return coeficients

def generate_training_data(numberofdata):
    """
    This function generates random data for the training data
    :param numberofdata: Number of points
    :return: The training data generated
    """
    x = []
    y = []
    for i in range(0, numberofdata):
        x.append(random.uniform(0, 6))
    x.sort()
    noise = np.random.normal(0, 0.1, numberofdata)
    pure = np.sin(x)
    y = pure + noise
    return x,y

def generate_test_data(numberofdata):
    """
    This function generates the test data set
    :param numberofdata: number of points in the data set
    :return: data set (input,target)
    """
    x = []
    y = []
    for i in range(0, numberofdata):
        x.append(random.uniform(0, 6))
    x.sort()
    noise = np.random.normal(0, 0.05, numberofdata)
    pure = np.sin(x)
    y = pure + noise
    return x,y

def make_predictions(x,coef):
    """
    Makes predictions for new data points
    :param x: new data point
    :param coef: coefficients of the predicted function
    :return: predicted value
    """
    w = []
    for ix in range(len(x)):
        q = 0
        count = 0
        for a in coef:
            q = q + a*x[ix]**count
            count = count + 1
        w.append(q)
    return w

def calculate_error(y,w):
    """
    Calculates the error
    :param y: target value
    :param w: predicted value
    :return: error
    """
    n = len(y)
    error = 0
    for i in range(len(y)):
        a = (y[i]-w[i])
        error = error + pow(a,2)
    return error

def plot_error(x,y, x_test, y_test):
    """
    Plots error
    :param x: input
    :param y: target value
    :param x_test: x test
    :param y_test: y test
    :return:
    """
    erms = []
    erms_test = []
    axis_x = [0,1,2,3,4,5,6,7,8,9]
    plt.ylim(0, 1)
    for i in axis_x:
        coef = mle_matrix_calculation_moorse(i,x,y)
        w = make_predictions(x, coef)
        w_test = make_predictions(x_test, coef)
        error = calculate_error(y, w)
        error = math.sqrt(error/len(x))
        error_test = calculate_error(y_test, w_test)
        error_test = math.sqrt(error_test/len(x))
        erms.append(error)
        erms_test.append(error_test)

    plt.plot(axis_x,erms,'o-', label="Training data")
    plt.plot(axis_x, erms_test, 'o-', label="Test data")
    plt.legend(bbox_to_anchor=(1, 1), loc='upper right', borderaxespad=0.)
    plt.title('Error function for different M')
    plt.xlabel('Coefficient M')
    plt.ylabel('Erms')
    plt.show()

def plot_error_regularization(x,y,x_test,y_test, degree):
    """
    Prints the regularization error
    :param x: x
    :param y: y
    :param x_test: x test
    :param y_test: y test
    :param degree: degree of the polynomial function
    :return:
    """
    erms = []
    erms_test = []
    axis_x = np.linspace(-40, 0, num=20)
    plt.ylim(0, 1)
    for i in axis_x:
        coef = calculate_matrix_regulatization(degree,x,y,pow(np.e,i))
        w = make_predictions(x, coef)
        w_test = make_predictions(x_test, coef)
        error = calculate_error(y, w)
        error = math.sqrt(error/len(x))
        error_test = calculate_error(y_test, w_test)
        error_test = math.sqrt(error_test/len(x))
        erms.append(error)
        erms_test.append(error_test)

    plt.plot(axis_x,erms, label="Training data")
    plt.plot(axis_x, erms_test, label="Test data")
    plt.legend(bbox_to_anchor=(1, 1), loc='upper right', borderaxespad=0.)
    plt.title('Error function for different regularization parameters')
    plt.xlabel('ln(lambda)')
    plt.ylabel('Erms')
    plt.show()

def calculate_matrix_regulatization(degree,x,y,l):
    """
    Calculates the matrix for regularization using Moorse-Penrose genaralized inverse
    :param degree: degree of the polynomial function
    :param x: x
    :param y: y
    :param l: lambda
    :return:
    """
    matrix = []
    row = []

    for a in x:
        for n in range(degree+1):
            row.append(pow(a, n))
        matrix.append(row)
        row = []
    tmatrix = np.transpose(matrix)
    mmatrix = np.dot(tmatrix, matrix)
    identity = np.identity((len(mmatrix)))
    sum = l*identity + mmatrix
    imatrix = np.linalg.inv(sum)
    nmatrix = np.dot(imatrix,tmatrix)
    coef = np.dot(nmatrix,y)
    return coef

if __name__ == "__main__":

    number_of_data = 100
    print("Welcome to the regression program!")
    x,y = generate_training_data(number_of_data)
    grades = [0,1,3,9]
    for grade in grades:
        coef = mle_matrix_calculation_moorse(int(grade),x,y)
        w = make_predictions(x,coef)
        error = calculate_error(y,w)
        plot(x,y,coef)
    x_test,y_test = generate_test_data(number_of_data)
    plot_error(x,y, x_test,y_test)
    l = pow(np.e,-18)
    grade = 9
    coef_regu = calculate_matrix_regulatization(int(grade),x,y, 1)
    plot(x,y,coef_regu)
    coef_regu = calculate_matrix_regulatization(int(grade), x, y, l)
    plot(x, y, coef_regu)
    plot_error_regularization(x,y,x_test,y_test, int(grade))