'''
	Vectorised Linear regression 
'''
import numpy as np
import matplotlib.pyplot as plt  
import time

def read_data():
	x_train = np.loadtxt('../data/x_train.csv')
	y_train = np.loadtxt('../data/y_train.csv')

	#preprocessing the data
	x_train = np.log(x_train)

	return x_train, y_train, x_test, y_test

def get_gradient(X, Y, W):
	'''
		compute the gradient of the loss function
	'''
	dW = np.array([0,0])
	dW = dW.reshape(2,1)
	n = float(len(x_train))
	
	dW = (2.0/n) * (X.T.dot(X).dot(W) - X.T.dot(Y))

	return dW


def create_model(X_train, Y_train, epochs = 10000, learning_rate = 0.001):
	'''
		fit a linear model such that

			y = w1 + x * w2

		where w1 and w2 are the parameters of the model which we want to learn
	'''
	
	W = np.array([0,0])
	W = W.reshape(2,1)


	print('model is running... ')
	for i in range(epochs):
		#get the gradients 
		dW = get_gradient(X_train, Y_train, W)

		#update the weights
		W = W - learning_rate * dW
	
	return W[0][0], W[1][0], W


def main():
	x_train, y_train = read_data()
	Y_train = y_train.reshape(y_train.shape[0],1)

	#Converting the array to a matrix and adding a column with 1
	temp = x_train.reshape(x_train.shape[0],len(x_train[0]))
	X_train = np.ones((temp.shape[0],temp.shape[1]+1))
	X_train[:,1:] = temp
	
	#Run the model
	start_time = time.time()
	w1,w2, W = create_model(X_train, Y_train)
	print(time.time() - start_time)

	#compute the model prediction
	y_pred = X_train.dot(W)

	#plot the graph
	plt.scatter(x_train, y_train, color='red')
	plt.plot(x_train, y_pred, color='blue')
	plt.show()

if __name__ == '__main__':

	





