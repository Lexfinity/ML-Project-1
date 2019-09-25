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

    
    def mean(self, X):
        y = X.iloc[:,-1]
        sum0 = 0
        sum1 = 0
        

        for i in range(len(y.index)):
            counter0 = 0
            counter1 = 0
            if y[i] == 0:
                w = X.iloc[i,:]
                for j in range(len(w.index)):
                    sum0 += w[j]
                counter0 = counter0 + 1
             
            if y[i] == 1:
                w = X.iloc[i,:]
                for j in range(len(w.index)):
                    sum1 += w[j]
                counter1 = counter1 + 1    


            tup1 = (sum0/counter0, sum1/counter1)
            return tup1
                





    

    def calculate_covariance_matrix(self, X):
            Y = (X- X.mean(axis=0))
            n_samples = np.shape(X)[0]
            covariance_matrix = Y.T.dot(Y)
            return np.array(covariance_matrix, dtype=float)
    

    def fit(self, sample, y):
        sample_0 = sample[y == 0]
        sample_1 = sample[y == 1]
        n_samples = np.shape(sample)[0]

        cVariance_0 = calculate_covariance_matrix(sample_0)
        cVariance_1 = calculate_covariance_matrix(sample_1)

        cVariance = cVariance_0 + cVariance_1
        cVariance = cVariance/(n_samples - 2)

