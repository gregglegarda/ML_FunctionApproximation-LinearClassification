import numpy as np
import matplotlib.pyplot as plt
import math




######INPUT LAYER (CHANGE THE INPUTS)
alpha = 0.01                        #Learning rate
range1= -2
range2= 2
stepsize = 0.1
neuronslayer1 = 10                  #Hidden layer
neuronslayer2 = 1                  #Output layer


p = np.arange(range1,range2,stepsize)#p = np.array([[-2],[-1.5],[-1],[-0.5],[0],[0.5],[1],[1.5],[2]]) #p = np.array([[1],[4],[1]]) #p = np.array([[1]])     ######test
p= p.reshape(p.shape[0],1)
print("THIS IS A ",p.shape[1],"-- S",neuronslayer1,"--",neuronslayer2," NETWORK")
print("WITH", p.shape[0], "INPUTS FROM",range1, "TO",range2)
print("AND ALPHA OF " , alpha,"\n")


######FUNCTION DEFINITIONS
def sigmoid(x):
    return 1 / (1 + math.exp(-x))
sigmoid_v = np.vectorize(sigmoid)

def t(x):
    return 1+ np.sin((np.pi/2.)*x)
t_v = np.vectorize(t)



######INITIALIZE WEIGHTS AND BIAS
W1 = np.random.uniform(-0.5,0.5, size= (p.shape[0],neuronslayer1,p.shape[1]))  #W1 = np.array([-0.27,-0.41]) ######test
b1 = np.random.uniform(-0.5,0.5, size= (p.shape[0],neuronslayer1,p.shape[1]))  #b1 = np.array([-0.48,-0.13]) ######test
W2 = np.random.uniform(-0.5,0.5, size= (p.shape[0],neuronslayer2,neuronslayer1)) #W2 = np.array([0.09,-0.17])   ######test
b2 = np.random.uniform(-0.5,0.5, size= (p.shape[0],neuronslayer2)) #b2 = np.array([0.48])         ######test


###################-----------EPOCH START-----------#################################
epoch = 1
while True:
    print("Epoch number:", epoch)                              ###############Checkpoint
    E =np.zeros(shape=(p.shape))
    
    ######HIDDEN LAYER (FIRST LAYER)
    x1=0
    a1 = np.zeros(shape=(p.shape[0],neuronslayer1,p.shape[1]))
    n1 = np.zeros(shape=(p.shape[0],neuronslayer1,p.shape[1]))
    for p1 in p:
        n1[x1] = np.dot(W1[x1], p[x1]).reshape(a1[x1].shape) + b1[x1]
        tempa = sigmoid_v(n1[x1])
        a1[x1] = tempa
        x1=x1+1
    #print("Output for the first layer (a1): ")                 ###############Checkpoint
    #print(a1)                                                  ###############Checkpoint
    
    
    
    ######OUTPUT LAYER (SECOND LAYER)
    x2=0
    a2 = np.zeros(shape=(p.shape))
    n2 = np.zeros(shape=(p.shape))
    for i in a1:
        n2[x2] = np.dot(W2[x2], a1[x2]).reshape(a2[x2].shape) + b2[x2]
        a2 = n2
        x2=x2+1
    #print("Output for the second layer (a2): ")    ###############Checkpoint
    #print(a2)                                      ###############Checkpoint
    
    
    
    ######FINDING THE ERROR
    x3=0
    for a,p1 in zip(a2,p):
        e = t_v(p1)-a
        E[x3]=e
        x3=x3+1
    #print("Errors for each output for the last layer (e): ")     ###############Checkpoint
    #print(E)                                                    ###############Checkpoint
    
    
    
    ######BACK PROPAGATION OF THE SENSITIVITIES
    #Last layer
    s2 = -2*(1)*E
    #print("Sensitivities for the last layer are (s2): ")    ###############Checkpoint
    #print(s2)                                               ###############Checkpoint
    
    #First layer
    s1= np.zeros(shape = (W2.shape[0],W2.shape[2],W2.shape[1]))
    F1n1 = np.zeros(shape = (s2.shape[0],neuronslayer1,neuronslayer1))
    W2s2 = np.zeros(shape = (W2.shape[0],W2.shape[2],W2.shape[1]))
    x4=0
    for i in W2:
        W2s2[x4] = np.transpose(W2[x4])*s2[x4]
        x4=x4+1
    j=0
    for i in F1n1:
        F1n1[j]=np.identity(neuronslayer1)
        F1n1[j]=F1n1[j]*(1-a1[j])*(a1[j])
        j=j+1
    dotF1n1 = np.dot(F1n1,np.ones((neuronslayer1,1)))
    
    x6=0
    for i in s1:
        s1[x6] = dotF1n1[x6]*W2s2[x6]
        x6=x6+1
    #print("Sensitivities for the first layer are (s1): ")    ###############Checkpoint
    #print(s1)                                                ###############Checkpoint
    
    
    #######WEIGHT AND BIAS UPDATES
    #loop for multiplying every s1 with every p (s1xp)
    x7=0
    s1xp=np.zeros(shape=s1.shape)
    for i in s1:
        s1xp[x7]=s1[x7]*p[x7]
        x7=x7+1
    
    #loop for multiplying every s2 with every a1 (s2a1)
    x8=0
    s2a1=np.zeros(shape=W2.shape)
    for i in s2:
        s2a1[x8]=s2[x8]*a1[x8].T
        x8=x8+1

    #actual updates
    W2 = W2-(alpha*s2a1)
    b2 = b2-(alpha*s2)
    W1 = W1-(alpha*s1xp)
    b1 = b1-(alpha*s1)

    #####checkpoint
    #print("New weight and bias updates are: ")
    #print("W2=")
    #print(W2)
    #print("b2=")
    #print(b2)
    #print("W1=")
    #print(W1)
    #print("b1=")
    #print(b1)

    ##########PLOTTING
    # target
    plt.plot(p, t(p), linestyle='--', marker='', color='b')
    #MLP
    plt.plot(p, a2, linestyle='-', marker='', color='r')
    plt.show()
    
    
    epoch=epoch+1
    if np.sum(np.abs(E)) == 0:
        break
###################-----------EPOCH FINISH-----------#################################





#Answers:
##different learning rates
#discusse convergence properties
#More neurons is a faster convergence.
#with 2 neurons with alpha of 0.1 it approximated the function at 18 epochs
#with 10 neurons with alpha of 0.1 it approximated the function at 6 epochs

#Bigger than 0.1 learning rates makes a slower convergence. Because when I increased alpha, it took longer.
#with 2 neurons with alpha of 0.5 it approximated the function at 29 epochs
#with 10 neurons with alpha of 0.5 it approximated the function at 10 epochs

#Smaller than 0.1 learning rates makes a slower convergence. Because when I decreased alpha, it took longer.
#with 2 neurons with alpha of 0.05 it approximated the function at  43 epochs
#with 10 neurons with alpha of 0.05 it approximated the function at  16 epochs
