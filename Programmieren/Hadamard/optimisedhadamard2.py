import numpy as np
import cmath as cm
import matplotlib.pyplot as plt
from array import *  

#define the used Parameters
N_t = 501
i = np.arange(0,N_t)
t_i = i/(N_t-1)
t_step = t_i[1]-t_i[0]
y_z = 0.01
O_max =10
N_m = 20
N_e = 1000
sigma_st=0.001
alpha = 0.1


#generate the starting values for our rho
r_x = np.random.normal(0,1)
r_y = np.random.normal(0,1)
r_z = np.random.normal(0,1)

l_n = np.sqrt(np.power(r_x,2)+np.power(r_y,2)+np.power(r_z,2))

rho_x,rho_y,rho_z,Fidelity = np.zeros(N_t),np.zeros(N_t),np.zeros(N_t),np.zeros(2)
rho_x[0] = r_x/l_n
rho_y[0] = r_y/l_n
rho_z[0] = r_z/l_n


rho_x_dot = lambda rho_x, rho_y, rho_z, h_x, h_y, gamma_z: 2*h_y*rho_z - 2*gamma_z*rho_x
rho_y_dot = lambda rho_x, rho_y, rho_z, h_x, h_y, gamma_z: -2*h_x*rho_z - 2*gamma_z*rho_y
rho_z_dot = lambda rho_x, rho_y, rho_z, h_x, h_y: 2*h_x*rho_y - 2*h_y*rho_x

F_list=np.zeros(N_e)
#update the parameter via


def theta_new(theta,alpha,F_e,F,epsilon):
    return theta + alpha*(F_e-F)/epsilon

#Fidelity
def F(rho_x0,rho_x1,rho_y0,rho_y1,rho_z0,rho_z1):
    return 0.5*(1+rho_z1*rho_x0+rho_x1*rho_z0-rho_y0*rho_y1)


h_x = np.ones(N_m)
h_y = np.ones(N_m)

#condition boundary for h_x and h_y
def max_h(h_x,h_y,sigma,N_m):
    for i in range(0,N_m):
     upper = np.sqrt(np.power(h_x[i],2)+np.power(h_y[i],2))/(0.5*sigma)
     max = (1+upper+np.absolute(1-upper))/2
     h_x[i]=h_x[i]/max
     h_y[i]=h_y[i]/max
    return h_x,h_y
h_x,h_y = max_h(h_x,h_y,O_max,N_m)


def h_t(h_,t,N_m):
    h = 0
    for i in range(0,N_m):
        h_plus = h_[i]*np.sin(np.pi*(i+1)*t)
        h = h + h_plus
    return h


for n in range(0,N_e):

 for i in range(0,N_t):
    H_x = sum([h_x[n]*np.sin(np.pi*n*t_i[i]) for n in range(N_m)])
    H_y = sum([h_y[n]*np.sin(np.pi*n*t_i[i]) for n in range(N_m)])
    H_x = H_x / max(1, 2*np.sqrt(H_x**2 + H_y**2)/O_max)
    H_y = H_y / max(1, 2*np.sqrt(H_x**2 + H_y**2)/O_max)
    rho_x[i] =rho_x[(i-1)]+3/2*(2*h_t(h_y,t_i[i-1],N_m)*rho_z[(i-1)]-2*y_z*rho_x[i-1])*t_step-1/2*(2*h_t(h_y,t_i[i-2],N_m)*rho_z[(i-2)]-2*y_z*rho_x[i-2])*t_step
    rho_y[i] =rho_y[(i-1)]+ 3/2*(-2*h_t(h_x,t_i[i-1],N_m)*rho_z[(i-1)]-2*y_z*rho_y[i-1])*t_step-1/2*(-2*h_t(h_x,t_i[i-2],N_m)*rho_z[(i-2)]-2*y_z*rho_y[i-2])*t_step
    rho_z[i] =rho_z[(i-1)]+ 3/2*(2*h_t(h_x,t_i[i-1],N_m)*rho_y[(i-1)]-(2*h_t(h_y,t_i[i-1],N_m))*rho_x[i-1])*t_step-1/2*(2*h_t(h_x,t_i[i-2],N_m)*rho_y[(i-2)]-(2*h_t(h_y,t_i[i-2],N_m))*rho_x[i-2])*t_step
    
    rho_x += rho_x_dot(rho_x, rho_y, rho_z, H_x, H_y, y_z) * t_step
    rho_y += rho_y_dot(rho_x, rho_y, rho_z, H_x, H_y, y_z) * t_step
    rho_z += rho_z_dot(rho_x, rho_y, rho_z, H_x, H_y) * t_step
 Fidelity[0]= F(rho_x[0],rho_x[N_t-1],rho_y[0],rho_y[N_t-1],rho_z[0],rho_z[N_t-1])
 F_list[n]=Fidelity[0]
 #now with the shiftet trainable parameters
 eps = np.random.normal(0,sigma_st)
 h_x_e = h_x + eps
 h_y_e = h_y + eps
 h_x_e,h_y_e = max_h(h_x_e,h_y_e,O_max,N_m)

 
 for i in range(2,N_t):
    rho_x[i] =rho_x[(i-1)]+3/2*(2*h_t(h_y_e,t_i[i-1],N_m)*rho_z[(i-1)]-2*y_z*rho_x[i-1])*t_step-1/2*(2*h_t(h_y_e,t_i[i-2],N_m)*rho_z[(i-2)]-2*y_z*rho_x[i-2])*t_step
    rho_y[i] =rho_y[(i-1)]+ 3/2*(-2*h_t(h_x_e,t_i[i-1],N_m)*rho_z[(i-1)]-2*y_z*rho_y[i-1])*t_step-1/2*(-2*h_t(h_x_e,t_i[i-2],N_m)*rho_z[(i-2)]-2*y_z*rho_y[i-2])*t_step
    rho_z[i] =rho_z[(i-1)]+ 3/2*(2*h_t(h_x_e,t_i[i-1],N_m)*rho_y[(i-1)]-(2*h_t(h_y_e,t_i[i-1],N_m))*rho_x[i-1])*t_step-1/2*(2*h_t(h_x_e,t_i[i-2],N_m)*rho_y[(i-2)]-(2*h_t(h_y_e,t_i[i-2],N_m))*rho_x[i-2])*t_step
 Fidelity[1]= F(rho_x[0],rho_x[N_t-1],rho_y[0],rho_y[N_t-1],rho_z[0],rho_z[N_t-1])
#we can now calculate the new theta
 
 h_x = h_x + alpha*(Fidelity[1]-Fidelity[0])/eps
 h_y = h_y + alpha*(Fidelity[1]-Fidelity[0])/eps
 #rexcaling to stay in our boundaries

 h_x,h_y = max_h(h_x,h_y,O_max,N_m)





###################################
######Plotting the results#########
###################################
x = np.arange(N_e)

# create plot
fig = plt.figure()
ax = plt.subplot(111)
ax.plot(x, F_list, label='Fidelity')
plt.title('Plot for 20 modes')
ax.legend()

# save plot to file
fig.savefig('plot20modes.png')
plt.clf()
################
#exrcise c)
optimal_hx = h_x
optimal_hy = h_y

rho_x[0] = 1
rho_y[0] = 0
rho_z[0] = 0
rho_x[1] = 1
rho_y[1] = 0
rho_z[1] = 0

for i in range(2,N_t):
    rho_x[i] =rho_x[(i-1)]+3/2*(2*h_t(h_y,t_i[i-1],N_m)*rho_z[(i-1)]-2*y_z*rho_x[i-1])*t_step-1/2*(2*h_t(h_y,t_i[i-2],N_m)*rho_z[(i-2)]-2*y_z*rho_x[i-2])*t_step
    rho_y[i] =rho_y[(i-1)]+ 3/2*(-2*h_t(h_x,t_i[i-1],N_m)*rho_z[(i-1)]-2*y_z*rho_y[i-1])*t_step-1/2*(-2*h_t(h_x,t_i[i-2],N_m)*rho_z[(i-2)]-2*y_z*rho_y[i-2])*t_step
    rho_z[i] =rho_z[(i-1)]+ 3/2*(2*h_t(h_x,t_i[i-1],N_m)*rho_y[(i-1)]-(2*h_t(h_y,t_i[i-1],N_m))*rho_x[i-1])*t_step-1/2*(2*h_t(h_x,t_i[i-2],N_m)*rho_y[(i-2)]-(2*h_t(h_y,t_i[i-2],N_m))*rho_x[i-2])*t_step
print("for state rho_x",rho_x[N_t-1],rho_y[N_t-1],rho_z[N_t-1])
x = np.arange(N_t)

# create plot
fig = plt.figure()
ax = plt.subplot(111)
ax.plot(x, rho_x, label='rho_x')
ax.plot(x, rho_y, label='rho_y')
ax.plot(x, rho_z, label='rho_z')
plt.title('Plot for rho_x trajectory')
ax.legend()

# save plot to file
fig.savefig('plottrajectoryx.png')

rho_x[0] = 0
rho_y[0] = 1
rho_z[0] = 0

rho_x[1] = 0
rho_y[1] = 1
rho_z[1] = 0

for i in range(2,N_t):
    rho_x[i] =rho_x[(i-1)]+3/2*(2*h_t(h_y,t_i[i-1],N_m)*rho_z[(i-1)]-2*y_z*rho_x[i-1])*t_step-1/2*(2*h_t(h_y,t_i[i-2],N_m)*rho_z[(i-2)]-2*y_z*rho_x[i-2])*t_step
    rho_y[i] =rho_y[(i-1)]+ 3/2*(-2*h_t(h_x,t_i[i-1],N_m)*rho_z[(i-1)]-2*y_z*rho_y[i-1])*t_step-1/2*(-2*h_t(h_x,t_i[i-2],N_m)*rho_z[(i-2)]-2*y_z*rho_y[i-2])*t_step
    rho_z[i] =rho_z[(i-1)]+ 3/2*(2*h_t(h_x,t_i[i-1],N_m)*rho_y[(i-1)]-(2*h_t(h_y,t_i[i-1],N_m))*rho_x[i-1])*t_step-1/2*(2*h_t(h_x,t_i[i-2],N_m)*rho_y[(i-2)]-(2*h_t(h_y,t_i[i-2],N_m))*rho_x[i-2])*t_step
print("for state rho_y",rho_x[N_t-1],rho_y[N_t-1],rho_z[N_t-1])

# create plot
fig = plt.figure()
ax = plt.subplot(111)
ax.plot(x, rho_x, label='rho_x')
ax.plot(x, rho_y, label='rho_y')
ax.plot(x, rho_z, label='rho_z')
plt.title('Plot for rho_y trajectory')
ax.legend()

# save plot to file
fig.savefig('plottrajectoryy.png')

rho_x[0] = 0
rho_y[0] = 0
rho_z[0] = 1

rho_x[1] = 0
rho_y[1] = 0
rho_z[1] = 1

for i in range(2,N_t):
    rho_x[i] =rho_x[(i-1)]+3/2*(2*h_t(h_y,t_i[i-1],N_m)*rho_z[(i-1)]-2*y_z*rho_x[i-1])*t_step-1/2*(2*h_t(h_y,t_i[i-2],N_m)*rho_z[(i-2)]-2*y_z*rho_x[i-2])*t_step
    rho_y[i] =rho_y[(i-1)]+ 3/2*(-2*h_t(h_x,t_i[i-1],N_m)*rho_z[(i-1)]-2*y_z*rho_y[i-1])*t_step-1/2*(-2*h_t(h_x,t_i[i-2],N_m)*rho_z[(i-2)]-2*y_z*rho_y[i-2])*t_step
    rho_z[i] =rho_z[(i-1)]+ 3/2*(2*h_t(h_x,t_i[i-1],N_m)*rho_y[(i-1)]-(2*h_t(h_y,t_i[i-1],N_m))*rho_x[i-1])*t_step-1/2*(2*h_t(h_x,t_i[i-2],N_m)*rho_y[(i-2)]-(2*h_t(h_y,t_i[i-2],N_m))*rho_x[i-2])*t_step

# create plot
fig = plt.figure()
ax = plt.subplot(111)
ax.plot(x, rho_x, label='rho_x')
ax.plot(x, rho_y, label='rho_y')
ax.plot(x, rho_z, label='rho_z')
plt.title('Plot for rho_z trajectory')
ax.legend()

# save plot to file
fig.savefig('plottrajectoryz.png')