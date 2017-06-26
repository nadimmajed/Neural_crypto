# -*- coding: utf-8 -*-
import numpy as np
from keras.models import Sequential
from keras.layers.core import Dense
from keras import initializers


class model:

    def __init__(self,hidden_unit,activation1,activation2,mean,stddev,training_data,target_data):
        self.hidden_unit = hidden_unit
        self.activation2 = activation2
        self.activation1 = activation1
        self.mean = mean
        self.stddev = stddev
        self.training_data = training_data
        self.target_data = target_data
            
    def without_bias(self,m):

        np.random.seed(1337+2*m)
        model = Sequential()
        model.add(Dense(self.hidden_unit, input_dim=8, 
                        activation=self.activation1,
                        kernel_initializer=initializers.RandomNormal
                        (mean=self.mean, stddev=self.stddev, seed=None)))
        model.add(Dense(8, activation=self.activation2))
        
        model.compile(loss='mean_squared_error',
                      optimizer='adam',
                      metrics=['accuracy'])
        
        model.fit(self.training_data, self.target_data, epochs=500, verbose=0)
        
        x= model.predict(self.training_data).round()
                    
        score =0
    # this is to calculate the accuracy because the accuracy of evaluate is giving a different value
        for i in range(0,256):
            for j in range(0,8):
                if x[i][j]!=self.target_data[i][j] :
                    score = score +1
    
        acc = 1-score/(256*8)
    # the loss value
        
        return  acc