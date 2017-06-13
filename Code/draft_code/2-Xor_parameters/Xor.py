def plot(s,j,accuracy,loss):
    """s = with bias or s = without bias"""
    from matplotlib import pyplot as plt
    plt.subplot(2,2,j)
    plt.ylabel('accuracy')
    plt.xlabel('nb_hidden_units')
    xlabel = [2,3,4,5]
    plt.plot(xlabel, accuracy, 'ro')
    plt.scatter(xlabel, accuracy)
    plt.ylim((0,1.25))
    plt.tight_layout()
    plt.title(s)
    # annotate the graph with loss values
    for i, txt in enumerate(loss):
        plt.annotate(txt,(xlabel[i],accuracy[i]))
    plt.show()
    
if __name__ =='__main__' :
    import numpy as np
    np.random.seed(1337)

    from math import sqrt
    activation2 = 'sigmoid'
    mean = 0
   # mean = 0.5
    stddev = sqrt(0.1)
    #stddev =1

    for j in range(0,2):
        if j ==0:
            activation1 = 'relu'
        else :
            activation1 = 'sigmoid'
        accuracy=[]
        loss = []
        accuracyB=[]
        lossB=[]
        for i in range(2,6):
            from Keras_Xor.model import model
            model = model(i,activation1,activation2,mean,stddev)
            acc, l,x = model.without_bias()
            accuracy.append(acc[0])
            loss.append(l[0])

            
            from Keras_Xor.modelB import modelB
            modelB= modelB(i,activation1,activation2,mean,stddev)
            accB, lB,xB = modelB.with_bias()
            accuracyB.append(accB[0])
            lossB.append(lB[0])
            print(i)
            print(activation1)
            print(xB)
            print(accB)

        

            
        plot(' (without_bias) '+'('+activation1+')' ,2*j+1,accuracy,loss)
        plot('(with bias)' +'('+activation1+')', 2*j+2, accuracyB, lossB)