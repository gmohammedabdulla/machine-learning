import numpy as np  

#binary class classification
class LogisticRegression:
	def __init__(self,learning_rate = 0.01, regularisation_parameter = 0.01):
		self.W = None
		self.b = None
		self.learning_rate = learning_rate
		self.regularisation_parameter = regularisation_parameter
		pass

	def fit(self, X, Y, epoch = 100):
		X = np.array(X)
		Y = np.array(Y).reshape(X.shape[0],1)
		self.W = np.random.randn(X.shape[1],1)
		self.b = 0
		for i in xrange(epoch):
			Y_hat = self.predict_proba(X)
			dW = -1 * X.T.dot(Y * (1-Y_hat) - (1-Y) * Y_hat)
			db = np.sum(-1 * (Y * (1-Y_hat) - (1-Y) * Y_hat))

			#print self.W.shape, dW.shape, Y.shape, Y_hat.shape
			self.W -= self.learning_rate * dW + 2 * self.regularisation_parameter * self.W 
			self.b -= self.learning_rate * db  

	def predict(self, X):
		if self.W ==  None:
			print "Cannot predict before training the model."
			return
		Y = X.dot(self.W) + self.b
		Y = self.sigmoid(Y)
		Y = [1 if z > 0.5 else 0 for z in Y ]
		return Y

	def predict_proba(self, X):
		if self.W ==  None:
			print "Cannot predict before training the model."
			return 
		Y = X.dot(self.W) + self.b
		return self.sigmoid(Y)

	def sigmoid(self, X):
		return 1.0 / (1 + np.exp(-X))
