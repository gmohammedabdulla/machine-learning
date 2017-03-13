'''
	Naive Linear regression using loops
'''
import numpy as np
import matplotlib.pyplot as plt  
import time

def read_data():
	'''
		Reading the data
	'''
	x_train = np.loadtxt('../data/x_train.csv')
	y_train = np.loadtxt('../data/y_train.csv')

	#preprocessing the data
	x_train = np.log(x_train)


	return x_train, y_train

def get_gradient(x_train, y_train, w1, w2):
	'''
		compute the gradient of the loss function
	'''
	dw1 = 0
	dw2 = 0
	n = float(len(x_train))
	
	for x,y in zip(x_train, y_train):
		dw1 += (2/n) * (y - w1 - x * w2) * (-1)
		dw2 += (2/n) * (y - w1 - x * w2) * (-1 * x)

	return dw1, dw2


def create_model(x_train, y_train, epochs = 10000, learning_rate = 0.001):
	'''
		fit a linear model such that

			y = w1 + x * w2

		where w1 and w2 are the parameters of the model which we want to learn
	'''	
	w1 = 0
	w2 = 0

	print('model is running... ')
	
	for i in range(epochs):
		#get the gradients 
		dw1 , dw2 = get_gradient(x_train, y_train, w1, w2)

		#update the weights 
		w1 = w1 - (learning_rate * dw1)
		w2 = w2 - (learning_rate * dw2)

	return w1, w2

def main():
	x_train, y_train = read_data()

	#create the model
	start_time = time.time()
	w1,w2 = create_model(x_train, y_train)
	print(time.time() - start_time)

	#calculate the predicted values using our model
	y_pred = []
	for x in x_train:
		y = w1 + x * w2
		y_pred.append(y)

	#plot the graph showing actual values and our prediction
	plt.scatter(x_train, y_train, color='red')
	plt.plot(x_train, y_pred, color='blue')
	plt.show()


if __name__ == '__main__':
	main()





