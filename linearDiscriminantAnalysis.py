import math
import numpy as np
import sys

class linearDiscriminantAnalysis:
    def __init__(self, input): 
    # output):
        self.input = input
        # self.output = output

    
    def NumberOfPositiveValues(self, x):
        num1 = np.count_nonzero(x.iloc[:,-1])
        total = x.shape[0]
        # row_count = sum(1 for row in x) 
        probOf1 = num1 / total
        print(probOf1)
        
    def NumberOfNegativeValues(self, x):
        num1 = x.shape[0] - np.count_nonzero(x.iloc[:,-1])
        total = x.shape[0]  
        # row_count = sum(1 for row in x) 
        probOf0 = num1 / total
        print(probOf0) 

    
    def split_data(self, X):
        y = X[:,-1]
        sum0 = 0
        sum1 = 0
        

        for i in range(y):
            if y[i] == 0:
                w = X[i,:]
                for j in range(w):
                    sum0 += w[j]
                    





    

    def calculate_covariance_matrix(X, Y=None):
        if Y is None:
            Y = X
            n_samples = np.shape(X)[0]
            covariance_matrix = (1/(n_samples-2)) * (X- X.mean(axis=0)).T.dot(Y-Y.mean(axis=0))
            return np.array(covariance_matrix, dtype=float)
    

    def fit(self, sample, y):
        sample_0 = sample[y == 0]
        sample_1 = sample[y == 1]

        cVariance_0 = calculate_covariance_matrix(sample_0)
        cVariance_1 = calculate_covariance_matrix(sample_1)

        cVariance = cVariance_0 + cVariance_1

