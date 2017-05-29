import numpy as np
from keras.models import Sequential
from keras.layers.core import Dense
from keras import initializers
np.random.seed(1337)

# the four different states of the XOR gate
training_data = np.array([[0,0],[0,1],[1,0],[1,1]], "float32")

# the four expected results in the same order
target_data = np.array([[0],[1],[1],[0]], "float32")

for j in range(0,2):
    if j ==0:
        activation = 'relu'
    else :
        activation = 'sigmoid'
    loss = []
    accuracy =[]
    # loop for the different hidden unit from 2 to 5
    for i in range(2,6):
        model = Sequential()
        model.add(Dense(i, input_dim=2, activation=activation,kernel_initializer=initializers.RandomNormal(mean=0.0, stddev=1, seed=None)))
        model.add(Dense(1, activation='sigmoid'))
        
        model.compile(loss='mean_squared_error',
                      optimizer='adam',
                      metrics=['accuracy'])
        
        model.fit(training_data, target_data, nb_epoch=500, verbose=0)
        
        x= model.predict(training_data).round()
        score =0
    # this is to calculate the accuracy because the accuracy of evaluate is giving a different value
        for i in range(0,4):
            if x.item(i)!=target_data.item(i):
                score = score +1
    
        acc = 1-score/4
        accuracy.append(acc)
        scores = model.evaluate(training_data, x, verbose=1)
    # the loss value
        loss.append("%.2f" % scores[0])
    
    
    
    lossB = []
    accuracyB =[]
    for i in range(2,6):
        model = Sequential()
        model.add(Dense(i, input_dim=2, activation=activation,kernel_initializer=initializers.RandomNormal(mean=0.0, stddev=1, seed=None),bias_initializer='one'))
        model.add(Dense(1, activation='sigmoid'))
        
        model.compile(loss='mean_squared_error',
                      optimizer='adam',
                      metrics=['accuracy'])
        
        model.fit(training_data, target_data, nb_epoch=500, verbose=0)
        
        x= model.predict(training_data).round()
        score =0
    # this is to calculate the accuracy because the accuracy of evaluate is giving a different value
        for i in range(0,4):
            if x.item(i)!=target_data.item(i):
                score = score +1
    
        acc = 1-score/4
        accuracyB.append(acc)
        scores = model.evaluate(training_data, x, verbose=1)
    # the loss value
        lossB.append("%.2f" % scores[0])
    
    
    
    from matplotlib import pyplot as plt
    
    plt.subplot(2,2,2*j+1)
    plt.ylabel('accuracy')
    plt.xlabel('nb_hidden_units(without bias)')
    xlabel = [2,3,4,5]
    plt.plot(xlabel, accuracy, 'ro')
    plt.axis([1,6,0,1.2])
    plt.scatter(xlabel, accuracy)
    # annotate the graph with loss values
    for i, txt in enumerate(loss):
        plt.annotate(txt,(xlabel[i],accuracy[i]))
      
    
    plt.subplot(2,2,2*j+2)
    #plt.ylabel('accuracy')
    plt.xlabel('nb_hidden_units(with bias)')
    xlabel = [2,3,4,5]
    plt.plot(xlabel, accuracyB, 'ro')
    plt.axis([1,6,0,1.2])
    plt.scatter(xlabel, accuracyB)
    # annotate the graph with loss values
    for i, txt in enumerate(lossB):
        plt.annotate(txt,(xlabel[i],accuracyB[i]))
    
    print("the activation function is",activation)
plt.show()

#bias_initializer='one'
    














































## -*- coding: utf-8 -*-
#import numpy as np 
#from keras.models import Sequential
#from keras.layers.core import Activation, Dense
#from keras import initializers
#
## the four different states of the XOR gate
#training_data = np.array([[0,0],[0,1],[1,0],[1,1]],"float32")
#
## the four expected results in the same order
#target_data = np.array([[0],[1],[1],[0]],"float32")
#model = Sequential()
#
#
#
##start the loop for the number of hidden units
#acc = []
#loss = []
#for i in range(2,6):
#    model.add(Dense(i, input_dim=2,activation = 'relu', 
#                    kernel_initializer=initializers.RandomNormal(mean=0.0, stddev=1, seed=None)))
#    #model.add(Dense(5, input_dim=2,activation = 'relu'))
#    model.add(Dense(1, activation = 'sigmoid'))
#    
#    
#    
#    model.compile(loss = 'mean_squared_error',
#                  optimizer= 'adam',
#                  metrics =['binary_accuracy'])
#    
#    model.fit(training_data, target_data, epochs = 5, verbose = 2)
#    
#    Y_test=model.predict(training_data).round()
#
#    scores = model.evaluate(training_data, Y_test, verbose=1)
#    
#    
#    print('Test loss:', scores[0])
#    print('Test accuracy:', scores[1])
#    loss.append(scores[0])
#    acc.append(scores[1])
#    
#
#from matplotlib import pyplot as plt
#import numpy as np
#plt.xlabel('accuracy')
#plt.ylabel('nb_hidden_units')
#plt.plot([2,3,4,5], acc, 'ro')
#plt.axis([1,6,0,1.2])
##"plt.show()
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
##weights = model.layers[0].get_weights()[0]
##biases = model.layers[0].get_weights()[1]
##
##print(weights)
##print(biases)
#
