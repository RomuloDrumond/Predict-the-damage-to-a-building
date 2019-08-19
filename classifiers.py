# Creating a class to represent the classifier
import numpy as np
from numpy.linalg import norm

class NearestCentroid():
    def __init__(self):
        self.class_list = {}
        self.centroids = {}
    
    def fit(self, X, y):
        self.class_list = np.unique(y, axis=0) # list of classes
        
        # Basically calculates centroids per class
        self.centroids = np.zeros((len(self.class_list), X.shape[1])) # each row is a centroid
        for i in range(len(self.class_list)): # for each class we evaluate its centroid
            temp = np.where(y==self.class_list[i][0])[0]
            self.centroids[i,:] = np.mean(X[temp],axis=0)
            
            
    def predict(self, X):
        N = len(X)
        y_pred = np.zeros(N)
        for i in range(N):
            # find closest
            dist = np.inf
            closest = None
            for j in range(len(self.class_list)):
                temp = X[i] - self.centroids[j]
                new_dist = np.dot(temp,temp)
                if new_dist < dist:
                    dist = new_dist
                    closest = self.class_list[j][0]
            
            y_pred[i] = closest
        
        return y_pred
    
    
    
from numpy.linalg import inv
from numpy import dot, matmul

class QuadraticGaussianClassifier():
    def __init__(self, variant=None, Lambda=None):
        self.variant = variant
        self.Lambda = Lambda
        self.class_list = {}
        self.centroids = {}
        self.C_inv = {}
        
    
    def fit(self, X, y):
        self.class_list = np.unique(y, axis=0) # list of classes
        
        # Basically calculates centroids per class
        self.centroids = np.zeros((len(self.class_list), X.shape[1])) # each row is a centroid
        for i in range(len(self.class_list)): # for each class we evaluate its centroid
            temp = np.where(y==self.class_list[i][0])[0]
            self.centroids[i,:] = np.mean(X[temp],axis=0)
            
        # Covariance matrices evaluation
        C_list = [np.zeros((X.shape[1], X.shape[1]))]*len(self.class_list)
        for i in range(len(self.class_list)):
            # points of class class_list[i]:
            temp = X[np.where(y==self.class_list[i][0])[0]]
            for datapoint in temp:
                aux = np.asmatrix(datapoint - self.centroids[i])
                C_list[i] += np.matmul(aux.T,aux)
            C_list[i] /= len(temp) # divide by number of samples
        
        if self.variant is None:
            self.C_inv = [inv(C) for C in C_list]
        
        elif self.variant==1: # diagonal
            self.C_inv = [inv(np.diag(np.diag(C))) for C in C_list]
        
        elif self.variant==2: # pooled
            self.C_inv = inv(self.__C_pool(X, y, C_list))
        
        elif self.variant==3: # Friedman regularization
            self.C_inv = [np.zeros((X.shape[1],X.shape[1]))]*len(self.class_list)
            N = len(y) # total number of samples
            for i in range(len(self.class_list)): # for each class
                Ni = len(X[np.where(y==self.class_list[i][0])[0]]) # number of samples of class 'i'
                self.C_inv[i] = inv(
                    ((1-self.Lambda)*Ni*C_list[i] + self.Lambda*N*self.__C_pool(X, y, C_list)) /
                    ((1-self.Lambda)*Ni           + self.Lambda*N) 
                )
            
    def __C_pool(self, X, y, C_list):
        C_poll = np.zeros((X.shape[1],X.shape[1]))
        for i in range(len(self.class_list)):
            # C_poll = sum (Ni/N) * Ci
            C_poll += ( len(X[np.where(y==self.class_list[i][0])[0]]) / len(y) ) * C_list[i]
        
        return C_poll
            
    def predict(self, X):
        N = len(X)
        y_pred = np.zeros(N)
        for i in range(N): # for each sample
            # find closest
            dist = np.inf
            closest = None
            for j in range(len(self.class_list)): # calculate the distance from each class
                temp = np.asmatrix(X[i] - self.centroids[j])
                # On variant 2 we have only one covariance matrix, in the other cases we have one covariance 
                # matrix for each class:
                C_inv = self.C_inv if self.variant==2 else self.C_inv[j]
                new_dist = np.matmul(np.matmul(temp,C_inv), temp.T)
                if new_dist < dist:
                    dist = new_dist
                    closest = self.class_list[j][0]
            
            y_pred[i] = closest
                        
        return y_pred