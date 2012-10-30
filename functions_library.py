# -*- coding: utf-8 -*-
"""
Created on Tue Oct 30 13:23:10 2012

@author: mikhail
"""

from numpy import *
from random import gauss
from numpy.linalg import *
from pylab import *

#--------------------------------------------------------------------------------
# Generate signal
#--------------------------------------------------------------------------------

def y(i, z, a1, a2, A_c, A_s, F, stage):
    return dot(D(i,z,tau,stage),x(A_c,A_s,F,stage))+noise(i,z,a1,a2)


#--------------------------------------------------------------------------------
# Define main functions
#--------------------------------------------------------------------------------

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

def noise(i,z,a1,a2): # define noise
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
    Estimator = dot(K,y)
    if stage == 1:
        return 1/w*arctan(Estimator[1]/Estimator[0]), 1/(m*w)*sqrt(Estimator[1]**2+Estimator[0]**2)
    else:
        return arctan(K[0][-3]/K[0][-4]) # Here  K[-3] means that we take the element 3d from the end. K[-4] - 4th from the end.

