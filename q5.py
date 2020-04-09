#!/usr/bin/python
import numpy as np
from numpy import genfromtxt
import math
from sklearn.svm import LinearSVC
from sklearn.svm import SVC
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score
import pandas as pd
from sklearn import svm

class AuthorClassifier:

    vectorizer = TfidfVectorizer(stop_words="english")

    def __init__(self, C=3):
	    self.C = C

    def fit_train(self, train_data, train_label):
        self.X_train = train_data
        self.y_train = train_label

    def fit_test(self, test_data, test_label):
        self.X_test = test_data
        self.y_test = test_label


    def train(self, train_path):
        data = pd.read_csv(train_path)
        y=data['author']
        y=y.to_numpy()
        X1 = self.vectorizer.fit(data['text'])
        vectors = self.vectorizer.transform(data['text'])
        X=vectors

        self.fit_train(X, y)

    def predict(self, test_path):
        data = pd.read_csv(test_path)
        y=data['author']
        y=y.to_numpy()
        vectors = self.vectorizer.transform(data['text'])
        X=vectors

        self.fit_test(X, y)

        #LinearSVC
        clf_c1 = LinearSVC(C=1.0)
        clf_c1.fit(self.X_train, self.y_train)
        # clf = svm.SVC(kernel='linear', C=1, decision_function_shape='ovr')
        # clf.fit(self.X_train, self.y_train)
        # y_pred=clf.predict(self.X_test)
        y_pred = clf_c1.predict(self.X_test)
        accuracy_c1 = accuracy_score(self.y_test, y_pred)
        print("Accuracy: ",accuracy_c1*100)
        
        return y_pred