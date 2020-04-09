from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_squared_error


from q3 import Airfoil as ar
model3 = ar()
model3.train('./Datasets/q3/train.csv') # Path to the train.csv will be provided
prediction3 = model3.predict('./Datasets/q3/test.csv') # Path to the test.csv will be provided
# prediction3 should be Python 1-D List 
'''WE WILL CHECK THE MEAN SQUARED ERROR OF PREDICTIONS WITH THE GROUND TRUTH VALUES'''


from q4 import Weather as wr
model4 = wr()
model4.train('./Datasets/q4/train.csv') # Path to the train.csv will be provided 
prediction4 = model4.predict('./Datasets/q4/test.csv') # Path to the test.csv will be provided
# prediction4 should be Python 1-D List 
'''WE WILL CHECK THE MEAN SQUARED ERROR OF PREDICTIONS WITH THE GROUND TRUTH VALUES'''


from q5 import AuthorClassifier as ac
auth_classifier = ac()
auth_classifier.train('./Datasets_1/q5/train.csv') # Path to the train.csv will be provided
predictions = auth_classifier.predict('./Datasets_1/q5/test.csv') # Path to the test.csv will be provided
'''WE WILL CHECK THE PREDICTIONS WITH THE GROUND TRUTH LABELS'''


from q6 import Cluster as cl
cluster_algo = cl()
# You will be given path to a directory which has a list of documents. You need to return a list of cluster labels for those documents
predictions = cluster_algo.cluster('./Datasets/q6/') 
'''SCORE BASED ON THE ACCURACY OF CLUSTER LABELS'''
