#!/usr/bin/python3
import numpy as np
from numpy import genfromtxt
import math
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
from sklearn.metrics import r2_score

class Airfoil:

    def __init__(self, alpha=0.01):
	    self.alpha = alpha

    def fit_train(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train

    def fit_test(self, X_test, y_test):
        self.X_test = X_test
        self.y_test = y_test

    def assign_theta(self, theta):
        self.theta = theta

    def gradient_descent(self, X_train, y_train, theta, alpha):
        predictions = list()
        cost_list = list()
        cost_list.append(1e10)
        no_of_iterations=0
        m=len(self.y_train)
        while True:
            temp = np.dot(self.X_train, self.theta)
            error = temp - self.y_train
            predictions.append(temp)
            cost = 1/(2*m) * np.dot(error.T, error)
            cost_list.append(cost)
            self.theta = self.theta - (self.alpha * (1/m) * np.dot(self.X_train.T, error))
            no_of_iterations = no_of_iterations + 1
            if(cost_list[no_of_iterations-1]-cost_list[no_of_iterations] < 1e-9):
                break

        cost_list.pop(0)
        return predictions, cost_list, no_of_iterations

    def train(self, train_path):
        data = pd.read_csv(train_path, header = None)
        y=data[5]
        y=y.to_numpy()
        X=data.to_numpy()
        X=X[:,:5]
        scaler = MinMaxScaler()
        scaler.fit(X)
        X=scaler.transform(X)

        ones = np.ones([X.shape[0],1])
        X = np.concatenate((ones,X),axis=1)
        # print(X_train.shape)
        # print(X_test.shape)
        # ones = np.ones([X_test.shape[0],1])
        # X_test = np.concatenate((ones,X_test),axis=1)
        # print(X_test.shape)
        np.random.seed(10)
        theta = np.random.rand(X.shape[1])
        # print(y_train.shape)

        self.fit_train(X, y)
        self.assign_theta(theta)

    def predict(self, test_path):
        data = pd.read_csv(test_path, header = None)
        y=data[5]
        y=y.to_numpy()
        X=data.to_numpy()
        X=X[:,:5]
        scaler = MinMaxScaler()
        scaler.fit(X)
        X=scaler.transform(X)

        ones = np.ones([X.shape[0],1])
        X = np.concatenate((ones,X),axis=1)
        np.random.seed(10)
        theta = np.random.rand(X.shape[1])

        self.fit_test(X, y)
        
        prediction_list, cost_list, no_of_iterations = self.gradient_descent(self.X_train, self.y_train, self.theta, self.alpha)
        
        y_predicted=np.dot(self.X_test,self.theta)

        print(r2_score(self.y_test,y_predicted))
        return y_predicted
