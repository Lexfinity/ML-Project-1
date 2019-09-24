import numpy as np
import pandas as pd
import numbers
import decimal
import logisticRegression as lR
import linearDiscriminantAnalysis as lDA


#auxiliary functions
 
def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        return False

print("----------------TASK 1-----------------")

wine_data = pd.read_csv('winequality-red.csv',sep=";", header=None)
cancer_data = pd.read_csv('breast-cancer-wisconsin.data',sep=",", header=None)

 
#Task 1   --    Acquire, Preprocess, and Analyze the Data
 
#Importing data
 
#Replace this with just the name of file if on PC
wine_data = pd.read_csv('winequality-red.csv',sep=";", header=None)
cancer_data = pd.read_csv('breast-cancer-wisconsin.data',sep=",", header=None)


# wine_data = pd.read_csv('/Users/aaronsossin/Documents/Fall2019/COMP551/winequality-red.csv', sep=';',header=None)
# cancer_data = pd.read_csv('/Users/aaronsossin/Documents/Fall2019/COMP551/breast-cancer-wisconsin.data', sep=',',header=None)
 

def deleteMalformedRows(d):
    rowsToDelete = []
    co = 0
    for index, row in d.iterrows():
        for cell in row:
            if not is_number(cell):
                print(cell)
                rowsToDelete.append(co)
                break
        co = co + 1
    
    for r in rowsToDelete:
        d.drop([r], inplace = True)
    
    return d
 
wine_data = deleteMalformedRows(wine_data)
cancer_data = deleteMalformedRows(cancer_data)
 
def convertLastColumnToBinary(d):
    c = 0 #counter
    for num in d.iloc[:,-1]: #for each value in last column
        if (float(num) < 6.0):
            d.iat[c,-1] = 0 #update array
        else:
            d.iat[c,-1] = 1
        c = c + 1
    return d
 
wine_data = convertLastColumnToBinary(wine_data)
 
def convertToNum(a):
    for x in range(0, a.shape[0]):
        for y in range(0, a.shape[1]):
            a.iloc[x,y] = float(a.iloc[x,y])
 
 
 
convertToNum(wine_data)
convertToNum(cancer_data)

#Statistics
 
# num1 = np.count_nonzero(wine_data.iloc[:,-1])
# num0 = wine_data.shape[0] - num1
# percent0 = (num0) / (wine_data.shape[0]) * 100
# print(percent0)


 
# print("-------logistic regression--------------")
# var = lR.logisticRegression(wine_data.iloc[:,:-1], wine_data.iloc[:,-1], 5, 1)
# var.start(1.5)

print("---------linear discriminant analysis------------")
var = lDA.linearDiscriminantAnalysis(wine_data.iloc[:,:-1])
print("Prob of 1")
var.NumberOfPositiveValues(wine_data)
print("Prob of 0")
var.NumberOfNegativeValues(wine_data)



# class LogisticRegression:
#     def __init__(self, lr=0.01, num_iter=100000, fit_intercept=True, verbose=False):
#         self.lr = lr
#         self.num_iter = num_iter
#         self.fit_intercept = fit_intercept
    
#     def __add_intercept(self, X):
#         intercept = np.ones((X.shape[0], 1))
#         return np.concatenate((intercept, X), axis=1)
    
#     #This is correct so far
#     def __sigmoid(self, z):
#         return 1 / (1 + np.exp(-z))

#     weight= 0.5
#     z= np.dot(cancer_data,weight)

#     # def __loss(self, h, y):
#     #     return (-y * np.log(h) - (1 - y) * np.log(1 - h)).mean()
    
#     def fit(self, X, y):
#         if self.fit_intercept:
#             X = self.__add_intercept(X)
        
#         # weights initialization
#         self.theta = np.zeros(X.shape[1])
        
#         for i in range(self.num_iter):

#             z = np.dot(X, self.theta)
#             h = self.__sigmoid(z)
#             gradient = np.dot(X.T, (h - y)) / y.size
#             self.theta -= self.lr * gradient
            
#             if(self.verbose == True and i % 10000 == 0):
                
#                 #fix z equation
#                 z = np.dot(X, self.theta)


#                 h = self.__sigmoid(z)
#                 print(f'loss: {self.__loss(h, y)} \t')
    
#     def predict_prob(self, X):
#         if self.fit_intercept:
#             X = self.__add_intercept(X)
    
#         return self.__sigmoid(np.dot(X, self.theta))
    
#     def predict(self, X, threshold):
#         return self.predict_prob(X) >= threshold
