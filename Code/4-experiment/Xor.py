def plot(s,j,accuracy,loss):
    """s = with bias or s = without bias"""
    
#    from matplotlib import pyplot as plt
#    plt.figure(1)
#    fig1 = plt.figure(1)
#    fig1.canvas.set_window_title('XOR_2bit')
#    plt.subplot(2,2,j)
#    plt.ylabel('accuracy')
#    plt.xlabel('nb_hidden_units')
#    xlabel = [2,3,4,5]
#    plt.plot(xlabel, accuracy, 'ro')
#    plt.scatter(xlabel, accuracy)
#    plt.ylim((0,1.25))
#    plt.tight_layout()
#    plt.title(s)
#    # annotate the graph with loss values
##    for i, txt in enumerate(loss):
##        plt.annotate(txt,(xlabel[i],accuracy[i]))
#    fig1.show()
    
def plotB(s,j,accuracy,loss):
    """s = with bias or s = without bias"""
    from matplotlib import pyplot as plt
    plt.figure(2)
    fig2 = plt.figure(2)
    fig2.canvas.set_window_title('XOR_11')
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
#    for i, txt in enumerate(loss):
#        plt.annotate(txt,(xlabel[i],accuracy[i]))
    fig2.show()
    
if __name__ =='__main__' :
    import numpy as np
    np.random.seed(1337)

    from math import sqrt
    activation1 = 'relu'
    activation2 = 'sigmoid'
    j=0
    k=0



    
#    for mean in (0,0.5):
#        for stddev in (0.1,1):
#            accuracy=[]
#            loss = []
#
#            for i in range(2,6):
#                tot =0
#                for m in range(0,10):
#                    from model import model
#                    np.random.seed(1337+2*m)
#                    model = model(i,activation1,activation2,mean,sqrt(stddev))
#                    acc= model.without_bias(m)
#                    tot = tot + acc
#                    
#                tot = tot/10
#                accuracy.append(acc)        
#                
#
#            plot(' (without_bias) '+'('+str(mean)+',' +str(stddev)+')' ,j+1,accuracy,loss)
#            j=j+1
            
    for mean in (0,0.5):
       for stddev in (0.1,1):
            accuracyB=[]
            lossB=[]
            
            for i in range(2,6):
                from modelB import modelB
                np.random.seed(1337)
                modelB= modelB(i,activation1,activation2,mean,sqrt(stddev))
                accB, lB,xB = modelB.with_bias()
                accuracyB.append(accB[0])
                lossB.append(lB[0])
                
            plotB('(with bias)' +'('+str(mean)+',' +str(stddev)+')', k+1, accuracyB, lossB)
            k=k+1
