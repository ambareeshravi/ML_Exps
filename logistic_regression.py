'''
Author: Ambareesh Ravi
Date: 7-02-2020
Title: Linear Regression

Links and References:
    Datasets:
        -
        
    Papers:
        -
        
    Lectures, tutorials, articles and posts:
        - 
        
    Additional comments / notes:
        - 
'''

# Imports
import warnings
warnings.filterwarnings('ignore')
import numpy as np
import pickle as pkl
import matplotlib.pyplot as plt

from time import time
import seaborn as sb
from tqdm import tqdm
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, mean_squared_error


class stochaistic_gradient_descent:
    def __init__(self, J, dJ, learning_rate = 0.01, max_iterations = 1500, precision = 1e-4):
        self.learning_rate = learning_rate # how much to update
        self.max_iterations = max_iterations # when to stop
        self.precision = precision # how precise the values are to be
        self.J = J # loss function
        self.dJ = dJ # first derivative of the loss function
        self.loss_history = list()
        self.stop_count = 0
        
    def run_SGD(self, X, Y, theta):
        for i in range(self.max_iterations):
            prev_theta = theta
            theta -= self.learning_rate * self.dJ(X, Y, theta)
            if np.all([abs(pt - ct) < self.precision for pt, ct in zip(prev_theta, theta)]) < 1: self.stop_count += 1; break
            loss = self.J(theta, X, Y)
            self.loss_history.append(loss)
        return theta, self.loss_history

class LogisticRegression:
    def __init__(self, threshold = 0.5):
        '''
        Initializes parameters for logistic regression
        '''
        self.threshold = threshold

    def add_bias(self, X):
        '''
        Function to add bias provision to the input
        
        Args:
            X - input <np.array>
        Returns:
            np.array with bias provision added
        Exception:
            -
        '''
        # adds an extra 1 to the data points for accomodating bias to make the math easier
        return np.array([np.array(x.tolist() + [1.0]) for x in X]) # last element is for bias
    
    def sigmoid(self, z):
        '''
        Sigmoid function
        
        Args:
            z - <float> value / <np.array>
        Returns:
            sigmiod value as <float> / <np.array>
        Exception:
            -
        '''
        return 1/(1+ np.exp(-z))
        
    def calc_h(self, theta, X):
        '''
        Calculate hypothesis function
        
        Args:
            theta - weights <np.array>
            input - X <np.array>
        Returns:
            value of the hypothesis function <np.array>
        Exception:
            -
        '''
        # hypothesis function
        return self.sigmoid(np.dot(self.add_bias(X), theta))

    def J(self, theta, X, y):
        '''
        Calculates the cost / loss function
        
        Args:
            theta - weights <np.array>
            X - inputs <np.array>
        Returns:
            value <float>
        Exception:
            -
        '''
        # cost function
        Ycap = self.calc_h(theta, X)
        return (1/len(y)) * (-np.dot(y, np.log(Ycap)) - np.dot((np.ones_like(y) - y), np.log(np.ones_like(y)-Ycap)))
    
    def dJ(self, theta, X, y):
        '''
        Calculates the derivative of the cost / loss function
        
        Args:
            theta - weights <np.array>
            X - inputs <np.array>
        Returns:
            value <float>
        Exception:
            -
        '''
        # gradient
        return (1 / len(y)) * np.dot(self.add_bias(X).T, (self.calc_h(theta, X) - y))
    
    def predict(self, X):
        '''
        Predicts the value using the model
        
        Args:
            X - inputs <np.array>
        Returns:
            prediction as probability <float>
        Exception:
            -
        '''
        return np.array([1.0 if p >= self.threshold else 0.0 for p in self.calc_h(self.best_theta, X)])
        
    def plot_graph(self, loss_history):
        '''
        Plots the graph epochs vs loss value
        
        Args:
            loss_history - inputs <np.array>
        Returns:
            -
        Exception:
            -
        '''
        plt.plot(range(len(loss_history)), loss_history)
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.title("Convergence for Logistic Regression optimzied with SGD\n")
        plt.show()
        
    def fit(self, X, y):
        '''
        Function to fit data X, y to logistic regression model
        
        Args:
            X - inputs <np.array>
            y - labels <np.array>
        Returns:
            best value of trained parameters / weights <np.array>
        Exception:
            -
        '''
        self.X = X
        self.y = y
        self.theta = np.zeros(X[0].shape[-1] + 1) # extra 1 for the bias
        sgd = StochaisticGradientDescent(self.J, self.dJ, learning_rate = 5e-1)
        
        # Run below snippet to get a smoother convergence with higher batch size
        # self.best_theta, loss_history = sgd.run_SGD(self.theta, self.X, self.y, max_iterations = 500, batch_size=128)
        # self.plot_graph(loss_history)
        
        self.best_theta, loss_history = sgd.run_SGD(self.theta, self.X, self.y, max_iterations = 500)
        self.plot_graph(loss_history)
        return self.best_theta
    
        def plot_data_decision(self):
            '''
            Plots the decision boundaries on the data

            Args:
                -
            Returns:
                -
            Exception:
                -
            '''
            print()
            X, y = self.parse_data()
            t = (self.best_theta[:2] / np.max(X)).tolist() + [self.best_theta[-1]] # restoring the effect of normalization -> bias wont be affected
            x1 = np.linspace(0,100)
            x2 = [(-t[-1] - (x * t[0])) / t[1] for x in x1]
            classes = ['class 1', 'class 2']
            colours = ListedColormap(['r','b'])
            scatter = plt.scatter(X[:,0], X[::,1], c=y, cmap=colours)
            plt.legend(handles=scatter.legend_elements()[0], labels=classes)
            plt.plot(x1,x2, color = 'black')
            plt.title("Logistic regression data and decision boundary")
            plt.xlabel("x1")
            plt.ylabel("x2")
            plt.show()

if __name__ == '__main__':
    tr_X, tr_y, ts_X, ts_y = get_data() 
    log_reg = LogisticRegression()
    best_theta = log_reg.fit(tr_X, tr_y)
    print("Test accuracy: ", accuracy_score(log_reg.predict(ts_X), ts_y))
    log_reg.plot_data_decision()