# neural net written using just numpy. The training set used is MNIST though
# it can be changed to take in any arbitrary dataset in csv format, ignoring
# the sklearn MNIST data set that is imported
from sklearn.datasets import fetch_mldata
import numpy as np
from sklearn.model_selection import train_test_split

digits = fetch_mldata('MNIST original',data_home = 'datasets/')
learning_rate = .66

X_train,X_test,y_train,y_test = train_test_split(digits.data,digits.target,test_size=0.2,random_state=1)

def sigmoid(z):
    return(1 / (1 + np.exp(-z)) )

def sigmoidDerivative(z):
    return (np.exp(-z)) / ((np.exp(-z) + 1 )**2)

exampleNumber = X_train.shape[0]
assert(X_train.shape[1] == 784) # all images have 784 pixels
hiddenNeurons = 28 #can be changed arbitrarily
y_train = y_train.reshape(exampleNumber,1)
newY_train = np.zeros((10,exampleNumber))

#scale to be between 0 and 1, prevents sigmoid overflow
X_train = X_train / 255.0
X_test = X_test / 255.0

for i in range(exampleNumber): #reshape y_training
    valueOriginal = y_train[i][0]
    newY_train[int(valueOriginal)][i] = 1

for iteration in range(1000): # 1000 gradient descent iterations, can be changed arbitrarily
    print(iteration)
    W1 = np.random.randn(hiddenNeurons,784)
    B1 = np.random.randn(hiddenNeurons,exampleNumber)
    Z1 = np.dot(W1,X_train.T) + B1
    A1 = sigmoid(Z1)
    W2 = np.random.rand(10,hiddenNeurons)
    B2 = np.random.randn(10,exampleNumber)
    Z2 = np.dot(W2,A1) + B2
    A2 = sigmoid(Z2)
    #backprop starts
    dz2 = A2 - newY_train
    dw2 = (1/exampleNumber)*(np.dot(dz2,A1.T))
    db2 = (1/exampleNumber)*(np.sum(dz2,axis=1,keepdims=True))
    dz1 = np.dot(W2.T,dz2)*(sigmoidDerivative(Z1))
    dw1 = (1/exampleNumber)*(np.dot(dz1,X_train))
    db1 = (1/exampleNumber)*(np.sum(dz1,axis=1,keepdims=True))
    if(np.sum(dw1) < .1 and np.sum(dw2) < .1 and np.sum(db1) < .1 and np.sum(db2) < .1):
        break
#predict on test set
