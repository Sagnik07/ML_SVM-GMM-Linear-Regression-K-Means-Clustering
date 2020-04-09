#!/usr/bin/python3
import numpy as np
from numpy import genfromtxt
import math
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
from sklearn.metrics import r2_score

class Weather:

    def __init__(self, alpha=0.1):
	    self.alpha = alpha

    def fit_train(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train

    # def fit_test(self, X_test, y_test):
    #     self.X_test = X_test
    #     self.y_test = y_test

    def fit_test(self, X_test):
        self.X_test = X_test

    def assign_theta(self, theta):
        self.theta = theta

    def gradient_descent(self, X_train, y_train):
        predictions = []
        cost_list = []
        cost_list.append(1e10)
        no_of_iterations=0
        m=len(self.y_train)
        while no_of_iterations<=10000:
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
        return predictions, cost_list

    def train(self, train_path):
        data = pd.read_csv(train_path)
        y_train=data['Apparent Temperature (C)']
        data=data.drop(['Apparent Temperature (C)','Formatted Date','Daily Summary'],axis=1)
        X_train=data
        y_train=y_train.to_numpy()

        # print("X_train")
        # print(X_train)
        # print("y_train")
        # print(y_train)
        dummy = pd.get_dummies(X_train)
        X_train=dummy.to_numpy()
        scaler = MinMaxScaler()
        scaler.fit(X_train)
        X_train=scaler.transform(X_train)
        
        print(X_train.shape)
        ones = np.ones([X_train.shape[0],1])
        # shape1=X_train.shape[0]+1
        X_train = np.concatenate((ones,X_train),axis=1)
        print(X_train.shape)
        np.random.seed(10)
        theta = np.random.rand(X_train.shape[1])
        print(theta.shape)
        self.fit_train(X_train, y_train)
        self.assign_theta(theta)

    def predict(self, test_path):
        test_data = pd.read_csv(test_path)
        # y_test=test_data['Apparent Temperature (C)']
        # test_data=test_data.drop(['Apparent Temperature (C)','Formatted Date','Daily Summary'],axis=1)
        test_data=test_data.drop(['Formatted Date','Daily Summary'],axis=1)
        X_test=test_data

        # print("X_test")
        # print(X_test)
        # print("y_test")
        # print(y_test)
        # y_test=y_test.to_numpy()
        dummy = pd.get_dummies(X_test)
        X_test=dummy.to_numpy()
        scaler = MinMaxScaler()
        scaler.fit(X_test)
        X_test=scaler.transform(X_test)
        ones = np.ones([X_test.shape[0],1])
        X_test = np.concatenate((ones,X_test),axis=1)
        
        # self.fit_test(X_test, y_test)
        self.fit_test(X_test)

        prediction_list, cost_list = self.gradient_descent(self.X_train, self.y_train)
        
        y_predicted=np.dot(self.X_test,self.theta)

        # print(r2_score(self.y_test,y_predicted))
        return y_predicted
