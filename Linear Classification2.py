import numpy as np
import matplotlib.pyplot as plt

# 1
# initialize
p = np.array([[1, -2], [1, 4], [2, -2], [2, 4], [3, 1], [3, 3], [4, 1], [4, 3]])
t = np.array([0, 0, 0, 0, 1, 1, 1, 1])

# averag of the points

# target a
a0 = np.average(p[0:4, 0])
a1 = np.average(p[0:4, 1])
avea = np.array([a0, a1])

# target b
b0 = np.average(p[4:, 0])
b1 = np.average(p[4:, 1])
aveb = np.array([b0, b1])

# midpoint

mpx = np.average(np.array([avea[0], aveb[0]]))
mpy = np.average(np.array([avea[1], aveb[1]]))
mp = np.array([mpx, mpy])

# initialize random Weight and Bias
W = np.random.rand(1, 2)
#W = np.zeros((1, 2))
b = 1

while True:
    # goes through each element in the "p" and "t" array
    E = np.array([])
    for x, y in zip(p, t):
        n = np.dot(W, x.reshape(2, 1)) + b
        if n < 0:
            a = 0
        else:
            a = 1
        e = y - a
        W = W + (e * x)
        b = b + e
        # put each error in array
        E = np.append(E, e)
    if np.sum(np.abs(E)) == 0:
        break
print("bias is: ")
print(b)
print("weight is: ")
print(W)

# decision boundary
p1 = np.arange(-3, 7, 1)
p2 = (-1 * (W[0, 0] * p1) / W[0, 1]) - (b / W[0, 1])

# p2 = (-1*(b/W[0,1])/(b/W[0,0]))*p1+(-1*b/W[0,1])


# 2
# the error is zero for each of the inputs and corresponding targets. Thus, we can say the weights and bias is correct.


# 3PLOTTING
# points
plt.plot(p[0:4, 0], p[0:4, 1], linestyle=' ', marker='o', color='b')
plt.plot(p[4:, 0], p[4:, 1], linestyle=' ', marker='*', color='r')

# decision boundary
plt.plot(p1, p2, linestyle='--', color='black')

# plot the midpoints
plt.plot(avea[0], avea[1], marker='+', color='black')
plt.plot(aveb[0], aveb[1], marker='+', color='black')
plt.plot(mp[0], mp[1], marker='+', color='black')


plt.show()




