#!/Users/violeta/miniforge3/envs/tensorflow_ARM/bin/python
import matplotlib.pyplot as plt
import numpy as np
from scipy import integrate as integ
import matplotlib.animation as animation
import matplotlib
#import ternary
import random
import math
import warnings
warnings.filterwarnings('ignore')

def social_groups(X,t,g1,g2):
    
    X1 = X[0:3]
    X2 = X[3:]

    A = np.array([[1, -1, 1], [-1, -1, -1], [1, -1, 0]])
    B = np.array([[-1, -1, -1], [-1, 1, 1], [-1, 1, 0]])

    sum1 =  X1[0] + X1[1] + X1[2]
    sum2 = X2[0] + X2[1] + X2[2]

    x1 = X1[0]/sum1
    y1 = X1[1]/sum1
    z1 = X1[2]/sum1

    x2 = X2[0]/sum2
    y2 = X2[1]/sum2
    z2 = X2[2]/sum2
    
    x1fitness = g1 * np.matmul(A, X1)[0] + g2 * np.matmul(B, X2)[0]
    y1fitness = g1 * np.matmul(A, X1)[1] + g2 * np.matmul(B, X2)[1]
    z1fitness = g1 * np.matmul(A, X1)[2] + g2 * np.matmul(B, X2)[2]

    x2fitness = g2 * np.matmul(A, X2)[0] + g1 * np.matmul(B, X1)[0]
    y2fitness = g2 * np.matmul(A, X2)[1] + g1 * np.matmul(B, X1)[1]
    z2fitness = g2 * np.matmul(A, X2)[2] + g1 * np.matmul(B, X1)[2]

    avfitnessg1 = x1*x1fitness + y1*y1fitness + z1*z1fitness
    avfitnessg2 = x2*x2fitness + y2*y2fitness + z2*z2fitness

    # Change in strategy frequencies:
    x1dot = x1*(x1fitness-avfitnessg1)
    y1dot = y1*(y1fitness-avfitnessg1)
    z1dot = z1*(z1fitness-avfitnessg1)
    x2dot = x2*(x2fitness-avfitnessg2)
    y2dot = y2*(y2fitness-avfitnessg2)
    z2dot = z2*(z2fitness-avfitnessg2)

    Xdot = np.array([x1dot,y1dot,z1dot,x2dot,y2dot,z2dot])    
    return Xdot

# Evolution of ethnic preferences:
x01 = np.ones(3,)
x02 = np.ones(3,)
#x01 = [0.1,0.1,0.8]
#x02 = [0.1,0.1,0.8]
x0 = np.concatenate((x01 / np.sum(x01), x02/np.sum(x02)))
x0 = x0.reshape(6,)

# for loop on different values of g1 and g2
g1 = np.arange(0,1,0.02)
g2 = 1 - g1

tend = 150 
tstep = 0.01 
t = np.arange(0,tend,tstep)
finalx = np.zeros((len(g1),6))
matplotlib.rc('font', size=16)


for i in range(len(g1)):

    x = integ.odeint(social_groups, x0, t, args=(g1[i],g2[i]))
    finalx[i,:] = x[-1,:]

# Plotting the final strategy frequencies:
fig, axs = plt.subplots(2)
axs[0].plot(g1, finalx[:,0], label=f'1+', color='b',linewidth=2)
axs[0].plot(g1, finalx[:,1], label=f'1-', color='b',linewidth=2, linestyle='--')
axs[0].plot(g1, finalx[:,2], label=f'1o', color='b',linewidth=2, linestyle=':')
axs[1].plot(g1, finalx[:,3], label=f'2+', color='r',linewidth=2)
axs[1].plot(g1, finalx[:,4], label=f'2-', color='r',linewidth=2, linestyle='--')
axs[1].plot(g1, finalx[:,5], label=f'2o', color='r',linewidth=2, linestyle=':')

axs[0].legend()
axs[1].legend()

fig.text(0.5, 0.01, 'group 1 percentage', ha='center')
fig.text(0, 0.5, 'final strategy frequencies', va='center', rotation='vertical')
fig.savefig('figures/gcenter.png')
print(fig)