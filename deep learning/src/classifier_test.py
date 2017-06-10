import numpy as np  
#from binary_logistic_regression import LogisticRegression
#from multiclass_logistic_regression import LogisticRegression
from neural_network import NeuralNetwork
from sklearn.utils import shuffle
#from sklearn.linear_model import LogisticRegression



l = 500
X1 = np.random.randn(l, 2) + np.array([0,2])
X2 = np.random.randn(l, 2) + np.array([0,-2])

Y = np.array([0]*l + [1]*l)
X = np.vstack([X1, X2])

X, Y = shuffle(X, Y)

'''
lr = LogisticRegression()
lr.fit(X, Y)
'''

nn = NeuralNetwork()
nn.fit(X,Y)
Y_hat = nn.predict(X)
print Y[:10], Y_hat[:10]
correct = sum([1 if a==b else 0 for a,b in zip(Y, Y_hat)])
print correct  * 1.0 /  len(Y)



