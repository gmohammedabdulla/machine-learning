import numpy as np  


class LogisticRegression:
	def __init__(self, learning_rate=0.01, regularisation_factor=0.01):
		self.W = None
		self.b = None
		self.learning_rate = learning_rate
		self.regularisation_factor = regularisation_factor

	def predict(self, X):
		if self.W == None:
			print 'Cannot Predict before training.'
			return 
		Z = X.dot(self.W) + self.b
		Z -= np.max(Z, axis=1, keepdims=True)
		expZ = np.exp(Z)
		Y = expZ / np.sum(expZ, axis = 1, keepdims=True)
		return np.argmax(Y, axis=1)

	def predict_proba(self, X):
		if self.W == None:
			print 'Cannot Predict before training.'
			return 
		Z = X.dot(self.W) + self.b
		Z -= np.max(Z, axis=1, keepdims=True)
		expZ = np.exp(Z)
		return expZ / np.sum(expZ, axis = 1, keepdims=True)

	def fit(self, X, Y, epoch=100):
		K = len(set(Y))
		X = np.array(X)
		D = X.shape[1]
		T = np.zeros((X.shape[0],K))
		for i in range(len(Y)):
			T[i,Y[i]] = 1
		
		self.W = np.random.randn(D,K)
		self.b = np.zeros(K)

		for i in range(epoch):
			Y_hat = self.predict_proba(X)
			cost = self.cost_fuction(Y_hat, T)
			print 'In epoch', i+1, 'Cost =',cost

			#Calculate the gradient
			dW = X.T.dot(Y_hat - T)
			db = np.sum(Y_hat - T, axis=0)

			#update the weight and bias
			self.W -= self.learning_rate * dW
			self.b -= self.learning_rate * db

	def cost_fuction(self,Y, T):
		return -1 * np.sum(T * np.log(Y))



