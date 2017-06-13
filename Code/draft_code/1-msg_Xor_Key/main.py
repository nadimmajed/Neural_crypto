#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May  8 11:11:09 2017

@author: nadim
"""
import numpy as np
import gen_cypher
import message

message = message('lorem.txt')
list = message.list_of_messages()

X=[]
Y=[]

key = np.random.randint(2,size=(16,))

for k in range(0,len(list)):
    cypher = gen_cypher(list[k],key)
    Y.append(cypher.generate_cypher())
    X.append(cypher.generate_bit())
    
print(cypher.bitlist_to_s())
 

from keras.models import Sequential
from keras.layers import Dense

X= np.array(X)
Y = np.array(Y)
seed = 7
np.random.seed(seed)

X_train= X[:900]
Y_train= Y[:900]
X_test= X[900:]
Y_test= Y[900:]
   # X_train = X_train.astype('float32')
#X_test = X_test.astype('float32')

# define baseline model
def baseline_model():
    	# create model
    	model = Sequential()
    	model.add(Dense(16, input_dim=16, kernel_initializer='normal', activation='sigmoid', bias_initializer='one'))
    	#model.add(Dense(num_classes, kernel_initializer='normal', activation='softmax'))
    	# Compile model
    	model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])
    	return model
#        
#epochs = 10
# build the model
model = baseline_model()
# Fit the model
model.fit(X_train, Y_train, validation_data=(X_test, Y_test), epochs=60000, batch_size=200, verbose=2)
# Final evaluation of the model

model.summary()

scores = model.evaluate(X_test, Y_test, verbose=1)
print("Baseline Error: %.2f%%" % (100-scores[1]*100))

weights = model.layers[0].get_weights()[0]
biases = model.layers[0].get_weights()[1]

print(weights)
print(biases)

print('Test score:', scores[0])
print('Test accuracy:', scores[1])
# -*- coding: utf-8 -*-

