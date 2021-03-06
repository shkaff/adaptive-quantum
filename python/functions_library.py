# -*- coding: utf-8 -*-
"""
Created on Tue Oct 30 13:23:10 2012

@author: mikhail
"""

from numpy import *
from random import gauss
from numpy.linalg import *
from pylab import *
import datetime
import os
"""
Matrices and arrays in python

Everywhere in the work we have tezors of 2nd order(matrices), so we use 2d
array for them. Even if we have vector, we use 2d array in order to have
the same dimensions for any variable.

We initialize them: a = array([[]])
When we want to add something to this array, we can choose, where to add:
from the right side or below. So, we add element in to ways:
a = append(a, data, axis = 1) -- from the right
a = append(a, data, 0) -- below
"""

#--------------------------------------------------------------------------------
# Define constants
#--------------------------------------------------------------------------------
procedure = 1 # we use variation method for innovation procedure

a = 1.0 # coupling constant
hbar = 1.0 # plank constant
m = 1.0 # mass of the oscillator
w = 1.0 # eigenfrequency of the oscillator

# signal
F = 50.1
tau = 0.9
A_c = 1/(m*w)*F*cos(w*tau)
A_s = 1/(m*w)*F*sin(w*tau)

initial_z = 0.9

# time step
N = 200
dt = 0.01
cycles = 1

#--------------------------------------------------------------------------------
# Define variables
#--------------------------------------------------------------------------------

#--------------------------------------------------------------------------------
# Generate signal
#--------------------------------------------------------------------------------
"""
Here y = Dx + n

y is number, x is column 2x1, D is vector 1x2, n is number

y = a1*cos(z) + a2*sin(z) + a/hbar * (signal + back action)
signal = F/(m*w)*sin(w*(t-tau)) = A_c*sin(w*t) - A_s*cos(w*t)
back action = a*integral_0^t(G(t'-t)*a1(t'))
"""

def y(i, z, a1, a2, A_c, A_s, F,BA, stage):
    return dot(D(i,z,tau,stage),x(A_c,A_s,F,stage))+noise(i,z,a1,a2,BA)

#--------------------------------------------------------------------------------
# Define main functions
#--------------------------------------------------------------------------------
"""
NB!

Here and everywhere: stage 1 is stage with estimation the arrival time,
stage 2 is sage with prediction the homodyne angle
"""

def D(i,z,tauEst,stage): # define measurement matrix
    if stage == 1:
        return array([[a/hbar*sin(w*dt*i)*sin(z[i]),-a/hbar*cos(w*dt*i)*sin(z[i])]])
    else:
        return array([[a/hbar*sin(w*(dt*i-tauEst))*sin(z)]])

def x(A_c,A_s,F,stage): # define signal
    if stage == 1:
        return array([[1/(m*w)*A_c], [1/(m*w)*A_s]])
    else:
        return F/(m*w)

def noise(i,z,a1,a2,BA): # define noise
    return n1(a1,i)*cos(z[i])+ n2(a2,i,BA)*sin(z[i])
    
#--------------------------------------------------------------------------------
# Define secondary functions
#--------------------------------------------------------------------------------
    
def n1(a1,i): # define noise in quadrature b1
    return a1[i]


def n2(a2,i,BA): # define noise in quadrature b2
    return a2[i]+a**2/hbar*BA[i]

def C(i,z): # homodyne measurement matrix
    return [cos(z[i]), sin(z[i])]

def G(t): # Green's function of the oscillator
    return (1/(m*w))*sin(w*t)
    
#--------------------------------------------------------------------------------
# Define covariations for noise
#--------------------------------------------------------------------------------

def noise_cov(i,j,z): # noise covariation matrix
    return dot(C(i,z),dot(nn(i,j),transpose(C(i,z))))
    
def nn(i,j): # covariation matrix of noise quadratures n1,n2
    return [[n1n1(i,j),n1n2(i,j)],[n2n1(i,j),n2n2(i,j)]]

def nn1(i,j,z):
    return n1n1(i,j)*cos(z[i]) + n2n1(i,j)*sin(z[i])

def nn2(i,j,z):
    return n1n2(i,j)*cos(z[i]) + n2n2(i,j)*sin(z[i])

def n1n(i,j,z):
    return n1n1(i,j)*cos(z[j]) + n1n2(i,j)*sin(z[j])

def n2n(i,j,z):
    return n2n1(i,j)*cos(z[j]) + n2n2(i,j)*sin(z[j])
    
def n1n1(i,j): # <b1(t_i),b1(t_j)> = 1/2 * \delta(t_i-t_j)
    if i==j:
        return 1.0/(2.0*dt)
    else:
        return 0.0
        
def n1n2(i,j): 
    if j>i:    
        return a**2.0/(2.0*hbar)*G(dt*j-dt*i)
    else:
        return 0.0

def n2n1(i,j): 
    if i>j:
        return a**2.0/(2.0*hbar)*G(dt*i-dt*j)
    else:
        return 0.0

def n2n2(i,j): 
    if i<j+1:  # we leave this term as i<j+1 because we need to take into account that at time i=j delta function will be non-zero.
        if i==j:
            return 1.0/(2*dt) + a**4/(4.0*hbar**2*m**2*w**2)*(dt*i*cos(w*(dt*i-dt*j))-1.0/w*cos(w*dt*i)*sin(w*dt*j)) 
        else:
            return a**4/(4.0*hbar**2*m**2*w**2)*(dt*i*cos(dt*i-dt*j)-1.0/w*cos(w*dt*j)*sin(w*dt*i))
    else:
        return a**4/(4.0*hbar**2*m**2*w**2)*(dt*j*cos(dt*j-dt*i)-1.0/w*cos(w*dt*i)*sin(w*dt*j)) 

#--------------------------------------------------------------------------------
# Define estimators
#--------------------------------------------------------------------------------
"""
#Here we will use the estimation for our signal:
#if the signal y = Dx+w
#Est_x = (D^T*N^{-1}*D)^{-1}*D^T*N^{-1}*y
#where C is covariation matrix of the noise (see Kay "Fundamentals of statistical signal processing:Estimation theory)
"""
def invert_matrix(i,z,old):
    V1 = array([[]])
    V2 = array([[]])
    V3 = array([[]])
    V4 = array([[]])
    V1 = old
    for j in xrange(0,i): # from 0 to i element, including i
        if j ==0:
            V2=append(V2, [[noise_cov(i,j,z)]], 1)
        else:
            V2=append(V2, [[noise_cov(i,j,z)]], 0)
        V3=append(V3, [[noise_cov(j,i,z)]], 1)
    V4 = append(V4,[[noise_cov(i,i,z)]],1)
    element = inv(V4 - dot(V3,dot(V1,V2)))
    inverse_1 = V1 + dot(V1,dot(V2,dot(element,dot(V3,V1))))
    inverse_2 = -dot(V1,dot(V2,element))
    inverse_3 = -dot(element,dot(V3,V1))
    inverse_4 = element 
    first_raw = append(inverse_1, inverse_2, 1)
    second_raw = append(inverse_3, inverse_4, 1)
    inverted_matrix = append(first_raw, second_raw, 0)    
    return inverted_matrix
    
def estimation_BLUE(invN,D,y,stage):
    transD = transpose(D)
    K = dot(dot(inv(dot(transD,dot(invN,D))),transD),invN) # filter
    
    if stage == 1:
        Estimator = dot(K,y)
        return abs(1/w*arctan(Estimator[1]/Estimator[0])), 1/(m*w)*sqrt(Estimator[1]**2+Estimator[0]**2), Estimator
    else:
        return arctan(K[0][-3]/K[0][-4]) # Here  K[-3] means that we take the element 3d from the end. K[-4] - 4th from the end.

def estimation_LMMSE(invY,xy,y,stage):
    K = dot(xy,invY)
    if stage == 1:
        Estimator = dot(K,y)
        return 1/w*arctan(Estimator[1]/Estimator[0]), 1/(m*w)*sqrt(Estimator[1]**2+Estimator[0]**2), Estimator
    else:
        return arctan(K[0][-3]/K[0][-4])
        

#--------------------------------------------------------------------------------
# Define innovation of homodyne angle
#--------------------------------------------------------------------------------
def variational(i,tau,T):
    zeta_constants = (4.0*w**2*m*hbar)/a**2
    zeta_numenator = sin(w*(dt*i-tau))
    zeta_denumenator = 2.0*w*(T - dt*i)*cos(w*(dt*i-tau)) + sin(w*(dt*i-tau))- sin(w*(2*T-dt*i-tau))
    return arctan(-zeta_constants*zeta_numenator/zeta_denumenator)
    
def adaptive(Ac, As, Ac_old, As_old):
    innovation_value = innovation_constant*dt*(Ac*Ac_old+As*As_old - Ac_old**2 - As_old**2)
    return innovation_value
    
#def filtering():
    
#--------------------------------------------------------------------------------
# Define estimation procedure
#--------------------------------------------------------------------------------
def back_action(N,dt):
    a1 = array([[]])
    a2 = array([[]])
    GG = zeros((N,N)) #back-action matrix
    for i in xrange(0,N):
        a1 = append(a1,[[gauss(0,sqrt(1.0/(2.0*dt)))]],1) # we use append(initial array, [[number]], axis)
        a2 = append(a2,[[gauss(0,sqrt(1.0/(2.0*dt)))]],1)
        for j in xrange(0,i+1): #fill the grin function matrix
            if i>j:
                GG[i][j] = G(dt*(i-j))
            else:
                GG[i][j] = 0.0   
    a1 = a1.T #transpose
    a2 = a2.T #transpose
    BA=dot(dot(GG,a1),dt) #calculate back-action matrix
    return BA, a1, a2

#-------
# Define innovation of the homodyne angle
#-------
def innovation_zeta(z,i,est_tau,Estimator,Estimator_old): 
    if procedure == 0: # no change
        z = append(z, [1.0])   
    if procedure == 1: # variation
        z = append(z, [variational(i+1,est_tau,dt*N)])
    if procedure == 2: # filtering
        z = 0
    if procedure == 3: #adaptive
        z = append(z, [z[i-1]+ adaptive(Estimator[0],Estimator[1],Estimator_old[0],Estimator_old[1])])
    return z
#-------
# Print results of the estimation 
#-------    
def printout(est_F,est_tau,estimation_F,estimation_tau,z,z_ideal):
    print est_F, 'estimation F', F, 'F'
    print est_tau, 'estimation tau', tau, 'tau'
    datet = datetime.datetime.now()
    fil = open('log.txt','a')
    fil.write('------------------------------------------------------------\n')
    fil.write('%s\n' % datet.strftime('%d-%m-%Y %H:%M'))
    fil.write('------------------------------------------------------------\n')
    fil.write('\t N=%s\n' % N)
    fil.write('\t dt=%s\n' % dt)
    fil.write('\t cycles=%s\n' % cycles)
    fil.write('\t estimation F=%s\n' % est_F)
    fil.write('\t F = %s\n' % F)
    fil.write('\t estimation tau = %s\n' % est_tau)
    fil.write('\t tau = %s\n' % tau)
    fil.close()
    
    directoryname = str(datet.strftime('%d-%m-%Y_%H:%M'))
    os.makedirs('images/%s' % directoryname)
    figure(0)
    plot(estimation_F)
    axhline(F, color = 'red')
    title(u'Estimation F')
    savefig('images/%s/estF.png' % directoryname)
    figure(1)
    plot(estimation_tau)
    axhline(tau, color = 'red')
    title(u'Estimation tau')
    savefig('images/%s/est_tau.png' % directoryname)
    figure(2)
    plot(z)
    plot(z_ideal)
    title(u'Homodyne angle')
    savefig('images/%s/z.png' % directoryname)
    show()
    
def mean_value(statistics_F,statistics_tau):
    F_mean = 0
    tau_mean = 0
    for i in xrange(0,cycles):
        F_mean += statistics_F[i]
        tau_mean += statistics_tau[i]
    F_mean = F_mean/cycles
    tau_mean = tau_mean/cycles
    error_F = F_mean-F
    error_tau = tau_mean-tau
    print F_mean, 'F_mean'
    print error_F, 'variation F'
    print error_tau, 'variation tau'
    datet = datetime.datetime.now()
    fil = open('log.txt','a')
    fil.write('------------------------------------------------------------\n')
    fil.write('%s\n' % datet.strftime('%d-%m-%Y %H:%M'))
    fil.write('------------------------------------------------------------\n')
    fil.write('\t N=%s\n' % N)
    fil.write('\t dt=%s\n' % dt)
    fil.write('\t cycles=%s\n' % cycles)
    fil.write('\t estimation F=%s\n' % est_F)
    fil.write('\t error F = %s\n' % error_F)
    fil.write('\t estimation tau = %s\n' % est_tau)
    fil.write('\t error tau = %s\n' % error_tau)
    fil.close()
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    