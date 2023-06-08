import numpy as np
import pandas as pd
#define X size 


wea = pd.read_csv('weather.csv')
n=30
X = np.zeros((n,3))  #26 days data, 23 unique user id steps
wea=np.array(wea)
wea=wea[0:n]

#Accumulate only the weather data from the array
   #26 days data, column: high temp, low temp, precipitation
wea = np.asfarray(wea, float)
#replicate weather for all the steps data

X=wea
ones_x = np.ones((n,1))

#create the X_matrix with Ones in the first column followed by weather data in the next
X_ = np.concatenate((ones_x,X),axis=1)

#load the steps data
steps = pd.read_csv('steps_count.csv')	
steps=np.array(steps)
steps=steps[0:n]
#create a matrix of just steps data, removing the IDs
y=steps
y = np.asfarray(y, float)


#define learning rate
alpha = 0.0001
#define the total number of iterations 
iters = 1000
#define initial weights
theta = np.array([[1.0,1.0,1.0,1.0]])

#compute the mean squared error
def computecost(X,y,theta):
    inner = np.power((np.matmul(X,theta.T)-y),2)
    return np.sum(inner)/(2*len(X))


#compute the gradient and update the cost
def gradientDescent(X,y,theta,alpha,iters):
    for i  in range(iters):
        theta = theta - (alpha/len(X)) * np.sum((X@theta.T-y)*X,axis=0)
        cost = computecost(X,y,theta)
        print(theta,cost)
    return (theta,cost)

g,cost=gradientDescent(X_,y,theta,alpha,iters)
print(g,cost)

old=[1725.18550495, 41.73728844, 16.29751196, 553.50462998]
input=[1.0, 60.0,55.0,0.1]

pred=np.dot(g,input)
# print("Predicted on 60,55,0.1 with custom Theta: ",pred)
predold=np.dot(old,input)
# print("Predicted on 60,55,0.1 with old theta: ",predold)


def getNewTheta():
	return g

def getOldTheta():
	return old

def getNewPrediction(input):
	pred=np.dot(g,input)
	return pred

def getOldPrediction(input):
	predold=np.dot(old,input)
	return predold


def predictRow(row):
    weat = pd.read_csv('weather.csv')
    stepc = pd.read_csv('steps_count.csv')  
    weat=np.array(weat)
    stepc=np.array(stepc)
    weat=weat[row]
    stepc=stepc[row]
    input=[1.0,weat[0],weat[1],weat[2]]
    print("Actual: ",stepc)
    print("Predicted: ",getNewPrediction(input))




