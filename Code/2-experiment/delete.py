# -*- coding: utf-8 -*-
import numpy as np
np.random.seed(1337)

from math import sqrt
activation1 = 'relu'
activation2 = 'sigmoid'
accuracy=[]
loss = []
mean =0
stddev = 1
from model import model
model = model(5,activation1,activation2,mean,sqrt(stddev))
acc, l,x = model.without_bias()
print(acc)
print(l)
print(x)
print("******************")
