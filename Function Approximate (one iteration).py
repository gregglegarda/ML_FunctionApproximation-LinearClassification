import numpy as np
import matplotlib.pyplot as plt
import math

######FUNCTION DEFINITIONS
def sigmoid(x):
    return 1 / (1 + math.exp(-x))
sigmoid_v = np.vectorize(sigmoid)

def t(x):
    return 1+ np.sin((np.pi/4.)*x)
t_v = np.vectorize(t)


######INPUT LAYER
alpha = 0.1
p = np.array([[-2],[-1.5],[-1],[-0.5],[0],[0.5],[1],[1.5],[2]]) #p = np.array([[1],[4],[1]]) #p = np.array([[1]]) #test
neuronslayer1 = 2
neuronslayer2 = p.shape[1]
print("THIS IS A ",p.shape[1],"-- S",neuronslayer1,"--",neuronslayer2," NETWORK")
print("WITH", p.shape[0], "INPUTS")
print("AND ALPHA OF " , alpha,"\n")


######HIDDEN LAYER (FIRST LAYER)
W1 = np.random.uniform(-0.5,0.5, neuronslayer1)  #W1 = np.array([-0.27,-0.41]) #test
b1 = np.random.uniform(-0.5,0.5, neuronslayer1)  #b1 = np.array([-0.48,-0.13]) #test
x1=0
a1 = np.zeros(shape=(p.shape[0],W1.shape[0]))
W1 = W1.reshape(neuronslayer1,p.shape[1])
b1 = b1.reshape(neuronslayer1,p.shape[1])
for p1 in p:
    n1 = np.dot(W1, p1) + b1.reshape(p.shape[1],neuronslayer1)
    tempa = sigmoid_v(n1)
    a1[x1] = tempa
    x1=x1+1
a1 = a1.reshape(p.shape[0],W1.shape[0],p.shape[1])
print("Output for the first layer (a1): ")
print(a1)


######OUTPUT LAYER (SECOND LAYER)
W2 = np.random.uniform(-0.5,0.5, neuronslayer1) #W2 = np.array([0.09,-0.17]) #test
b2 = np.random.uniform(-0.5,0.5, neuronslayer2) #b2 = np.array([0.48]) #test
x2=0
a2 = np.zeros(shape=(p.shape))
for i in a1:
    n2 = np.dot(W2,a1) + b2
    #print("b2 is:", b2)
    #print("W2 is:", W2)
    #print("a1 is:", a1)
    a2 = n2
    x2=x2+1
print("Output for the second layer (a2): ")
print(a2)


######FINDING THE ERROR
E =np.zeros(shape=(a2.shape))
x3=0
for a2,p2 in zip(a2,p):
    e = t_v(p2)-a2
    E[x3]=e
    x3=x3+1

print("Errors for each output for the last layer (e): ")
print(E)


######BACK PROPAGATION OF THE SENSITIVITIES
#(LAST LAYER)
s2 = -2*(1)*E
print("Sensitivities for the last layer are (s2): ")
print(s2)

#(FIRST LAYER)
F1n1 = np.zeros(shape = (s2.shape[0],neuronslayer1,neuronslayer1))
s1part2 = np.transpose(W2)*s2
s1part2 = s1part2.reshape(s2.shape[0],neuronslayer1,1)
j=0
for i in F1n1:
    F1n1[j]=np.identity(neuronslayer1)
    F1n1[j]=F1n1[j]*(1-a1[j])*(a1[j])
    j=j+1
s1part1 = np.dot(F1n1,np.ones((neuronslayer1,1)))

s1 = s1part1*s1part2
print("Sensitivities for the first layer are (s1): ")
print(s1)

print("W2=")
print(W2)
print("b2=")
print(b2)
print("W1=")
print(W1)
print("b1=")
print(b1)


#WEIGHT AND BIAS UPDATES
#loop for multiplying every s with every p (sxp)
v=0
sxp=np.zeros(shape=s1.shape)
for i in s1:
    sxp[v]=s1[v]*p[v]
    v=v+1
#actual updates
W2 = W2-(alpha*s2)*a1.reshape(p.shape[0],W1.shape[0])
b2 = b2-(alpha*s2)
W1 = W1-(alpha*sxp)
b1 = b1-(alpha*s1)



print("New weight and bias updates are: ")
print("W2=")
print(W2)
print("b2=")
print(b2)
print("W1=")
print(W1)
print("b1=")
print(b1)




#PLOTTING
# target
plt.plot(p, t(p), linestyle='-', marker='', color='b')


#plt.plot(p[4:, 0], p[4:, 1], linestyle=' ', marker='*', color='r')

# decision boundary
#plt.plot(p1, p2, linestyle='--', color='black')

# plot the midpoints
#plt.plot(avea[0], avea[1], marker='+', color='black')
#plt.plot(aveb[0], aveb[1], marker='+', color='black')
#plt.plot(mp[0], mp[1], marker='+', color='black')
# reminders
##different learning rates
#discusse convergence properties


plt.show()


