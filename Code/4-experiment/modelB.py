# -*- coding: utf-8 -*-
import numpy as np
from keras.models import Sequential
from keras.layers.core import Dense
from keras import initializers

import random
def truthtable (n):
  if n < 1:
    return [[]]
  subtable = truthtable(n-1)
  return [ row + [v] for row in subtable for v in [0,1] ]

training_data = truthtable (2)
x=int(random.uniform(0,len(training_data)))
w = []
print(training_data)
print(training_data[x])
for i in range(len(training_data)):
    w.append([training_data[i][0]^training_data[x][0],
              training_data[i][1]^training_data[x][1]])
    
target_data = w

class modelB:

    def __init__(self,hidden_unit,activation1,activation2,mean,stddev):
        self.hidden_unit = hidden_unit
        self.activation2 = activation2
        self.activation1 = activation1
        self.mean = mean
        self.stddev = stddev
        
    def with_bias(self,i):

        # the four different states of the XOR gate

        np.random.seed(1337+2*i)
        model = Sequential()
        model.add(Dense(self.hidden_unit, input_dim=2, activation=self.activation1,kernel_initializer=initializers.RandomNormal(mean=self.mean, stddev=self.stddev, seed=None),bias_initializer='one'))
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
    # the loss value
        
        return  acc