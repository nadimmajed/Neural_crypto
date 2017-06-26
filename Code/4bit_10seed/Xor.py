def plot(s,j,accuracy,std):
    """s = with bias or s = without bias"""
    
    from matplotlib import pyplot as plt
    plt.figure(1)
    fig1 = plt.figure(1)
    fig1.canvas.set_window_title('XOR_4bit')
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
    for i, txt in enumerate(std):
        plt.annotate(txt,(xlabel[i],accuracy[i]))
    fig1.show()

    
    
def plotB(s,j,accuracy,std):
    """s = with bias or s = without bias"""
    from matplotlib import pyplot as plt
    plt.figure(2)
    fig2 = plt.figure(2)
    fig2.canvas.set_window_title('XOR_4bit')
    plt.subplot(2,2,j)
    plt.ylabel('accuracy')
    plt.xlabel('nb_hidden_units')
    xlabel = [2,3,4,5]
    plt.plot(xlabel, accuracy, 'ro')
    plt.scatter(xlabel, accuracy)
    plt.ylim((0,1.25))
    plt.tight_layout()
    plt.title(s)
#     annotate the graph with loss values
    for i, txt in enumerate(std):
        plt.annotate(txt,(xlabel[i],accuracy[i]))
    fig2.show()
    
if __name__ =='__main__' :
    
    import random
    def truthtable (n):
        if n < 1:
            return [[]]
        subtable = truthtable(n-1)
        return [ row + [v] for row in subtable for v in [0,1] ]

    training_data = truthtable (4)
    x=int(random.uniform(0,len(training_data)))
    target_data = []
    print(training_data)
    print(training_data[x])
    key=[0, 1, 0, 0]

    for i in range(len(training_data)):
        target_data.append([training_data[i][0]^training_data[x][0],
                  training_data[i][1]^training_data[x][1],
                               training_data[i][2]^training_data[x][2],
                                    training_data[i][3]^training_data[x][3]])
    
    
    
    
    import numpy as np
    np.random.seed(1337)

    from math import sqrt
    activation1 = 'relu'
    activation2 = 'sigmoid'
    j=0
    k=0



    
    for mean in (0,0.5):
        for stddev in (0.1,1):
            accuracy=[]
            loss = []
            std = []

            for i in range(1,5):
                tot = []
                for m in range(0,10):
                    

                    from model import model
                    np.random.seed(1337+2*m)
                    model = model(2*i,activation1,activation2,mean,sqrt(stddev),training_data,target_data)
                    acc= model.without_bias(m)
                    tot.append(acc)
                accuracy.append(np.mean(tot))
                std.append(round(np.std(tot),2))
                    

            plot(' (without_bias) '+'('+str(mean)+',' +str(stddev)+')' ,j+1,accuracy,std)
            j=j+1
            
    for mean in (0,0.5):
       for stddev in (0.1,1):
            accuracyB=[]
            stdB=[]
            lossB=[]
            
            for o in range(1,5):
                tot = []
                for n in range(0,10):
                    from modelB import modelB
                    np.random.seed(1337+ 2*n)
                    modelB= modelB(2*o,activation1,activation2,mean,sqrt(stddev),training_data,target_data)
                    accB = modelB.with_bias(n)
                    tot.append(accB)
                accuracyB.append(np.mean(tot))
                stdB.append(round(np.std(tot),2))
                    
                
            plotB('(with bias)' +'('+str(mean)+',' +str(stddev)+')', k+1, accuracyB, stdB)
            k=k+1
