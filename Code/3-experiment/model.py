# -*- coding: utf-8 -*-
import numpy as np
from keras.models import Sequential
from keras.layers.core import Dense
from keras import initializers


class model:

    def __init__(self,hidden_unit,activation1,activation2,mean,stddev):
        self.hidden_unit = hidden_unit
        self.activation2 = activation2
        self.activation1 = activation1
        self.mean = mean
        self.stddev = stddev
            
    def without_bias(self):
        loss = []
        accuracy =[]
        # the four different states of the XOR gate
        training_data = np.array([[0,0],[0,1],[1,0],[1,1]], "float32")
        
        # the four expected results in the same order
        #target_data = np.array([[1,1],[1,0],[0,1],[0,0]], "float32")(1,1)
        #target_data = np.array([[1,0],[1,1],[0,0],[0,1]], "float32")(1,0)
        #target_data = np.array([[0,1],[0,0],[1,1],[1,0]], "float32")(0,1)
        target_data = np.array([[0,0],[0,1],[1,0],[1,1]], "float32")
        
        
        np.random.seed(1337)
        model = Sequential()
        model.add(Dense(self.hidden_unit, input_dim=2, 
                        activation=self.activation1,
                        kernel_initializer=initializers.RandomNormal
                        (mean=self.mean, stddev=self.stddev, seed=None)))
        model.add(Dense(2, activation=self.activation2))
        
        model.compile(loss='mean_squared_error',
                      optimizer='adam',
                      metrics=['accuracy'])
        
        model.fit(training_data, target_data, epochs=500, verbose=0)
        
        x= model.predict(training_data).round()
                    
        score =0
    # this is to calculate the accuracy because the accuracy of evaluate is giving a different value
        for i in range(0,4):
            for j in range(0,2):
                if x[i][j]!=target_data[i][j]:
                    score = score +1
    
        acc = 1-score/8
        accuracy.append(acc)
        scores = model.evaluate(training_data, x, verbose=0)
    # the loss value
        loss.append("%.2f" % scores[0])
        
        return  accuracy,loss,x
            