import pickle
import matplotlib.pyplot as plt
from matplotlib.axes._axes import _log as matplotlib_axes_logger 
matplotlib_axes_logger.setLevel('ERROR')
from matplotlib import style
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.stats import norm
np.random.seed(0)

def load(name):
    file = open(name,'rb')
    data = pickle.load(file)
    file.close()
    return data

def save(data,name):
    file = open(name, 'wb')
    pickle.dump(data,file)
    file.close()

class GMM1D:
    def __init__(self,X,iterations,initmean,initprob,initvariance):
        """initmean = [a,b,c] initprob=[1/3,1/3,1/3] initvariance=[d,e,f] """    
        self.iterations = iterations
        self.X = X
        self.mu = initmean
        self.pi = initprob
        self.var = initvariance
    
    """E step"""

    def calculate_prob(self,r):
        for c,g,p in zip(range(3),[norm(loc=self.mu[0],scale=self.var[0]),
                                       norm(loc=self.mu[1],scale=self.var[1]),
                                       norm(loc=self.mu[2],scale=self.var[2])],self.pi):
            r[:,c] = p*g.pdf(self.X)
        """
        Normalize the probabilities such that each row of r sums to 1 and weight it by mu_c == the fraction of points belonging to 
        cluster c
        """
        for loop in range(len(r)):
        	# Write code here
            r[loop] = r[loop]/(np.sum(self.pi)*np.sum(r,axis=1)[loop])
#             pass
        return r
    
    def plot(self,r):
        fig = plt.figure(figsize=(10,10))
        ax0 = fig.add_subplot(111)
        for i in range(len(r)):
            color=['r','g','b']
            ax0.scatter(self.X[i],0,c=color[int(i/60)],s=100)
        """Plot the gaussians"""
        for g,c in zip([norm(loc=self.mu[0],scale=self.var[0]).pdf(np.linspace(-20,20,num=60)),
                        norm(loc=self.mu[1],scale=self.var[1]).pdf(np.linspace(-20,20,num=60)),
                        norm(loc=self.mu[2],scale=self.var[2]).pdf(np.linspace(-20,20,num=60))],['r','g','b']):
            ax0.plot(np.linspace(-20,20,num=60),g,c=c)
    
    def run(self):
        
        for iter in range(self.iterations):

            """Create the array r with dimensionality nxK"""
            r = np.zeros((len(self.X),3))  

            """
            Probability for each datapoint x_i to belong to gaussian g 
            """
            r = self.calculate_prob(r)


            """Plot the data"""
            self.plot(r)
            
            """M-Step"""

            """calculate m_c"""
            m_c = []
            # write code here
            for i in range(len(r[0])):
                m_c1 = np.sum(r[:,i])
                m_c.append(m_c1)
            
            """calculate pi_c"""
            # write code here
            for k in range(len(m_c)):
                self.pi[k] = (m_c[k]/np.sum(m_c))
            
            """calculate mu_c"""
            # write code here
            self.mu = np.sum(self.X.reshape(len(self.X),1)*r,axis=0)/m_c
#             print(r[:,1].shape)
            """calculate var_c"""
            var_c = []
            #write code here
            for i in range(len(r[0])):
                var_c.append((1/m_c[i])*np.dot(((np.array(r[:,i]).reshape(180,1))*(self.X.reshape(len(self.X),1)-self.mu[i])).T,(self.X.reshape(len(self.X),1)-self.mu[i])))
            plt.show()


dict1 = load('Datasets/Question-2/dataset1.pkl')
dict2 = load('Datasets/Question-2/dataset2.pkl')
dict3 = load('Datasets/Question-2/dataset3.pkl')
data=[]
data=np.concatenate((dict1, dict2, dict3), axis=0)
mean1=np.mean(dict1)
mean2=np.mean(dict2)
mean3=np.mean(dict3)
var1=np.var(dict1)
var2=np.var(dict2)
var3=np.var(dict3)
X = np.squeeze(np.asarray(data))
g = GMM1D(X,10,[mean1,mean2,mean3],[1/3,1/3,1/3],[var1,var2,var3])
g.run()
