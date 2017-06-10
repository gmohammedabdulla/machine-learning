import numpy as np  

class NeuralNetwork:
	def __init__(self, learning_rate=0.01, regularisation_factor=0.01, hidden_layer = 5):
		self.W1 = None
		self.b1 = None
		self.W2 = None
		self.b2 = None
		self.learning_rate = learning_rate
		self.regularisation_factor = regularisation_factor
		self.hidden_layer = hidden_layer

	def predict(self, X):
		if self.W1 == None:
			print 'Cannot Predict before training.'
			return 
		Z = self.sigmoid(X.dot(self.W1) + self.b1)
		A = Z.dot(self.W2) + self.b2
		expA = np.exp(A)
		Y = expA / np.sum(expA, axis=1, keepdims=True)
		return np.argmax(Y, axis = 1)

	def predict_proba(self, X, return_Z = False):
		if self.W1 == None:
			print 'Cannot Predict before training.'
			return 
		Z = self.sigmoid(X.dot(self.W1) + self.b1)
		A = Z.dot(self.W2) + self.b2
		expA = np.exp(A)
		Y = expA / np.sum(expA, axis = 1,  keepdims=True)
		if return_Z == False:
			return Y
		return Y, Z

	def sigmoid(self, X):
		return 1.0 / (1 + np.exp(-X))

	def fit(self, X, Y, epochs=100):
		K = len(set(Y))
		X = np.array(X)
		D = X.shape[1]
		N = X.shape[0]
		H = self.hidden_layer
		T = np.zeros((N,K))
		for i in range(len(Y)):
			T[i,Y[i]] = 1

		self.W1 = np.random.randn(D,H)
		self.b1 = np.zeros(H)
		self.W2 = np.random.randn(H,K)
		self.b2 = np.zeros(K)

		for i in range(epochs):
			Y_hat, Z = self.predict_proba(X, return_Z = True)
			#print Y_hat.shape
			cost = self.cost_fuction(Y_hat, T)
			print 'In epoch', i+1, 'Cost =',cost

			dW2 = Z.T.dot(Y_hat - T)
			db2 = np.sum(Y_hat - T, axis=0)
			dZ = (Y_hat - T).dot(self.W2.T) * Z * (1-Z) # for relu activation function
														# replace Z * (1-Z) with derivative of 
														# relu fuction - (Z > 0).astype(int)
														# In forward pass replace sigmoid by relu 
			dW1 = X.T.dot(dZ)
			db1 = np.sum(dZ, axis=0)

			self.W2 -= self.learning_rate * dW2
			self.b2 -= self.learning_rate * db2
			self.W1 -= self.learning_rate * dW1
			self.b1 -= self.learning_rate * db1


	def cost_fuction(self,Y, T):
		return -1 * np.sum(T * np.log(Y))



