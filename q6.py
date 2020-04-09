#!/usr/bin/python
import numpy as np
from numpy import genfromtxt
import math
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score
import pandas as pd
import os

class Cluster:

    def __init__(self, no_of_clusters=5):
	    self.no_of_clusters = no_of_clusters

    def fit_train(self, train_data):
        self.X = train_data
        
    def vectorizer(self, files):
        vectorizer = TfidfVectorizer(stop_words="english")
        vectors = vectorizer.fit_transform(files.values())
        # X=vectors
        feature_names = vectorizer.get_feature_names()
        # print(feature_names)
        # print(vectors.shape)
        dense = vectors.todense()
        denselist = dense.tolist()
        df = pd.DataFrame(denselist, columns=feature_names)
        # df
        X=df.to_numpy()
        self.fit_train(X)

    def euclidean_distance(self, x):
        return (np.sqrt(np.sum(x**2)))

    def clustering(self, centroids):
        i=0
        loop=0
        prev_centroids=centroids
        for loop in range(15):
            classes={}
            indexes={}
            begin=0

            for i in range(self.no_of_clusters):
                indexes[i] = []
                classes[i] = []

            for row in self.X:
                # for c in range(len(centers)):
                #   dists[c]=np.linalg.norm(row-centers[c])
                distances = [self.euclidean_distance(row-centroids[i]) for i in range(len(centroids))]
                # print(dists)
                minimum_distance = min(distances)
                m_index = distances.index(minimum_distance)
                # print(index)
                indexes[m_index].append(begin)
                classes[m_index].append(row)
                begin = begin + 1

            for i in range(len(centroids)):
                centroids[i] = np.mean(classes[i], axis = 0)
            
        return indexes


    def find_majority_label(self, label):
        return max(set(label), key = label.count) 

    def cluster(self, path):
        # path = '/content/drive/My Drive/Assignment-2_Dataset/Datasets/Question-6/dataset'
        file_name = ""
        files = {}
        for a,b,f in os.walk(path):
            for file in f:
                if ".txt" in file:
                    file_name = os.path.join(a,file)
                    f1 = open(file_name, 'rb')
                    file_data = f1.read().decode(errors="replace")
                    file_data = " ".join(file_data.split())
                    files[file_name] = file_data

        self.vectorizer(files)

        centroids=np.random.uniform(size=(self.no_of_clusters,self.X.shape[1]))
        for i in range(self.no_of_clusters):
            centroids[i] = centroids[i]/(np.linalg.norm(centroids[i]))
    
        indexes = {}
        indexes = self.clustering(centroids)


        i=0
        file_labels={}
        # print(files.keys())
        for k in files.keys():
            x=k.split("_")
            # print(x)
            file_labels[i]=int(x[1][0])
            i=i+1

        # print(file_labels)
        # print(len(file_labels))
        
        orig_labels={}
        for i in indexes.keys():
            for j in indexes[i]:
                orig_labels[j]=file_labels[j]


        pred_label_of_cluster={}
        for loop in range(5):
            li=[]
            for i in indexes[loop]:
                li.append(file_labels[i])
            pred_label_of_cluster[loop]=self.find_majority_label(li)


        pred_label={}
        for i in indexes.keys():
            for j in indexes[i]:
                pred_label[j]=pred_label_of_cluster[i]

        y_true=orig_labels.values()
        y_pred=pred_label.values()
        y_true=list(y_true)
        y_pred=list(y_pred)
        print(accuracy_score(y_true,y_pred))
        return y_pred
