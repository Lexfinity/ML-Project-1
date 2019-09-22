import math
import numpy as np
 
class logisticRegression:
 
    def __init__(self, input, output, learningRate, descents):
        self.input = input
        self.output = output
        self.learningRate = learningRate
        self.descents = descents
        self.numberOfInputs = input.shape[0]
 
    def sigma(self,x):
        output = 1 / (1 + (math.e ** (-(x))))
        return output
 
    def updateFunction(self,x,y,w,n):
        print("update function")
        w = np.array(w)
        
        cur = [0.0] * len(w)
        counter = 0
        l = [0.0] * len(w)
        for z in range(0,n):
            try:
                l = np.multiply(x.iloc[z,:],(y.iloc[z] - self.sigma(np.dot(w.T,x.iloc[z,:]))))
            except: 
                counter = counter + 1
            cur = np.add(cur,l)
        updated = w + np.multiply(self.learningRate,cur)
        return updated
    
    def predict(self, x, y, w):
        print(x)
        sum = 0
        for z in range(len(w)):
            sum = sum + w[z] * x.iloc[z]
        a = sum
        print(a)
        probability = self.sigma(a)
        print(probability)
        
       
    
    def start(self, initialWeight):
        iterations = 5
        w = [initialWeight]
        for d in range(0,len(self.input.iloc[0,:])-1):
            w.append(initialWeight)
        for c in range(0,5): #screw iterations for now. 
            w = self.updateFunction(self.input, self.output, w, self.numberOfInputs)
        # print(w)
        for e in range(0,len(w)):
            w[e] = math.log(-w[e])
        self.predict(self.input.iloc[0,:], self.output.iloc[0], w)
