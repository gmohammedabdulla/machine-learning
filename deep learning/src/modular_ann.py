import numpy as np  

def feed_forward(X, W1, b1 , W2, b2):
	#sigmoid activation
	#Z = 1.0 / ( 1 + np.exp(-X.dot(W1) - b1))

	#relu activation
	Z = X.dot(W1) + b1
	Z = Z * ((Z>0).astype(int))

	A = Z.dot(W2) + b2
	expA = np.exp(A)
	Y = expA / expA.sum(axis=1, keepdims=True)
	return Y, Z

def derivative_l2(Y, T, Z, W2):
	dW2 = Z.T.dot(Y - T)
	db2 = (Y - T).sum(axis=0)
	dZ = (Y - T).dot(W2.T)
	return dW2, db2, dZ

def derivative_l1(dZ, Z, X):
	#sigmoid backprop
	#dZ = dZ * Z * (1-Z)

	#relu backprop
	dZ = dZ * ((Z>0).astype(int))

	dW1 = X.T.dot(dZ)
	db1 = (dZ).sum(axis=0)
	return dW1, db1

def one_hot_encode(Y):
	K = len(set(Y))
	N = len(Y)
	T = np.zeros((N,K))
	for i in range(N):
		T[i, Y[i]] = 1
	return T

def cost_function(Y,T):
	return np.sum(-1 * T * np.log(Y))


