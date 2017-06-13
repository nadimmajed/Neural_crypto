#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 30 17:36:19 2017

@author: nadim
"""
import numpy as np
np.random.seed(1337) # for reproducibility
from keras.models import Sequential
class gen_cypher():
    def __init__(self,message,key):
        self.message = message
        self.table = key
        self.bit_message = []
    
    def s_to_bitlist(self,s):
        ords=(ord(c) for c in s)
        shifts =(7,6,5,4,3,2,1,0)
        return [(o >> shift) & 1 for o in ords for shift in shifts]

    def generate_cypher(self):
        cypher = []
        
        for i in range(0,16):
            cypher.append(self.s_to_bitlist(self.message)[i]^self.table[i])
        return cypher
    

    def generate_bit(self):
        cy= []
        for i in range(0,16):
            cy.append(self.s_to_bitlist(self.message)[i])   
        return cy
    
    def test(self):
        test_c = []
        for i in range(0,16):
            test_c.append(self.generate_cypher()[i]^self.table[i])

        return test_c

    def bitlist_to_chars(self,t):
        bi = iter(t)
        bytes = zip(*(bi,)*8)
        shifts = (7, 6, 5, 4, 3, 2, 1, 0)
        for byte in bytes:
            yield chr(sum(bit << s for bit, s in zip(byte, shifts)))

    def bitlist_to_s(self):
        return ''.join(self.bitlist_to_chars(self.test()))

if __name__ == "__main__":
    X=[]
    Y=[]
    with open('lorem.txt') as f:
        lines = f.readlines()
    
   
        
    for i in range(0,len(lines)):
        lines[i]=lines[i].split(" ")
   
    
    list =[]
    for k in range(0,len(lines)):
        for j in range(0,len(lines[k])):
            list.append(lines[k][j])

    
    for i in range(0,len(list)):
        if len(list[i])<2 :
            list[i]= "bk" 

    key = np.random.randint(2,size=(16,))

    for k in range(0,len(list)):
        cypher = gen_cypher(list[k],key)
        Y.append(cypher.generate_cypher())
        X.append(cypher.generate_bit())
        
    print(cypher.bitlist_to_s())
     

    from keras.layers import Dense

    X= np.array(X)
    Y = np.array(Y)
    seed = 7
    np.random.seed(seed)
    print(len(X))
    a=1000
    b=1010
    X_train= X[:a]
    Y_train= Y[:a]
    X_test= X[a:b]
    Y_test= Y[a:b]
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
    model.fit(X_train, Y_train, validation_data=(X_test, Y_test), epochs=120, batch_size=20, verbose=2)
    # Final evaluation of the model
    
    model.summary()

    scores = model.evaluate(X_test, Y_test, verbose=1)
#    print("Baseline Error: %.2f%%" % (100-scores[1]*100))
    
    weights = model.layers[0].get_weights()[0]
    biases = model.layers[0].get_weights()[1]
    
    print(weights)
    print(biases)

    print('Test loss:', scores[0])
    print('Test accuracy:', scores[1])
