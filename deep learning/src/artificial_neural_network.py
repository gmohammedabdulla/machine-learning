import numpy as np  
#import matplotlib.pyplot as plt 


l = 500
X1 = np.random.randn(l, 2) + np.array([0,2])
X2 = np.random.randn(l, 2) + np.array([0,-2])
X3 = np.random.randn(l, 2) + np.array([-2,0])

Y = np.array([0]*l + [1]*l + [2]*l)
X = np.vstack([X1, X2, X3])

assert(len(X) == len(Y))

N = 5
K = 3
W1 = np.random.randn(2, N)
b1 = np.random.randn(N)
W2 = np.random.randn(N, K)
b2 = np.random.randn(K)


def plot_data(X,Y):
	plt.scatter(X[:,0], X[:,1], c=Y)
	plt.show()


def classification_rate(Y, P):
	return np.sum(Y == P) * 1.0/ len(Y)

def feed_forward(X, W1, b1 , W2, b2):
	#sigmoid activation
	#Z = 1.0 / ( 1 + np.exp(-X.dot(W1) - b1))

	#relu activation
	Z = X.dot(W1) + b1
	Z = Z * ((Z>0).astype(int))

	A = Z.dot(W2) + b2
	expA = np.exp(A)
	Y = expA / expA.sum(axis=1, keepdims=True)
	return Y

def backprop(X, T, W1, b1, W2, b2):
	alpha = 0.0001

	for q in range(1000):
		#sigmoid activation
		#Z = 1.0 / (1 + np.exp(-X.dot(W1) - b1))

		#relu activation
		Z = X.dot(W1) + b1
		Z = Z * ((Z>0).astype(int))

		A = Z.dot(W2) + b2
		
		expA = np.exp(A)
		Y = expA / expA.sum(axis=1, keepdims=True)

		dW1 = np.zeros(W1.shape)
		dW2 = np.zeros(W2.shape)
		db1 = np.zeros(b1.shape)
		db2 = np.zeros(b2.shape)
		dZ = np.zeros(Z.shape)

		#print W2.shape

		'''
		#Slow one
		for i in range(W2.shape[0]):
			for j in range(W2.shape[1]):
				for n in range(len(X)):
					dW2[i,j] += Z[n,i] * (Y[n,j] - T[n,j])
					dZ[n,i] += W2[i,j] * (Y[n,j] - T[n,j])

		for i in range(W2.shape[0]):
			for j in range(W2.shape[1]):
				for n in range(len(X)):
					db2[j] += Y[n,j] - T[n,j]

		for i in range(W1.shape[0]):
			for j in range(W1.shape[1]):
				for n in range(len(X)):
					dW1[i,j] += Z[n,j] * (1 - Z[n,j]) * X[n,i] * dZ[n,j]

		for i in range(W1.shape[0]):
			for j in range(W1.shape[1]):
				for n in range(len(X)):
					db1[j] = Z[n,j] * (1 - Z[n,j]) * dZ[n,j]
		'''

		# fast
		dW2 = Z.T.dot(Y - T)
		db2 = (Y - T).sum(axis=0)
		dZ = (Y - T).dot(W2.T)
		
		#sigmoid backprop
		#dZ = dZ * Z * (1-Z)

		#relu backprop
		dZ = dZ * ((Z>0).astype(int))

		dW1 = X.T.dot(dZ)
		db1 = (dZ).sum(axis=0)

		W1 = W1 - alpha * dW1
		W2 = W2 - alpha * dW2
		b1 = b1 - alpha * db1
		b2 = b2 - alpha * db2

	return W1, b1, W2, b2


T = np.zeros((l*3,K))
for i, y in enumerate(Y):
	T[i,y] = 1

W1, b1, W2, b2 = backprop(X, T, W1, b1, W2, b2)

P_Y_given_X = feed_forward(X, W1, b1, W2, b2)
P = np.argmax(P_Y_given_X, axis=1)

assert(len(P) == len(Y))

print classification_rate(Y, P)



