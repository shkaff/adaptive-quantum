# -*- coding: utf-8 -*-
"""
Created on Tue Oct 30 13:46:53 2012

@author: mikhail
"""
from numpy import *
from random import gauss
from numpy.linalg import *
from pylab import *
from functions_library import *



estimation_F = []
estimation_tau = []
statistics_F = []
statistics_tau = []


#################################################################################
#-------------------------------------------------------------------------------#
# Main cycle                                                                    #
#-------------------------------------------------------------------------------#
#################################################################################

for k in xrange(0, cycles):
    
#--------------------------------------------------------------------------------
# Define variables to change every loop
#--------------------------------------------------------------------------------
    

    z = array([[]])
    #initialize homodyne angle
    z=append(z, [[initial_z]]) #add frist element      
    z=append(z, [[initial_z]])
    
    y_signal = array([[]]) # vector of the signal
    meas_matrix = array([[]]) # measurement matrix D
    inv_mat=array([[]])
        
    BA, a1, a2 = back_action(N,dt) # generate back-action
    
    for i in xrange(0,N): 
        
        if i == 0:
            y_signal = append(y_signal,y(0, z, a1, a2, A_c, A_s, F, BA, 1),1)
            meas_matrix = append(meas_matrix,D(0,z,1,1),1)
            inv_mat = append(inv_mat,[[1/noise_cov(0,0,z)]],1)
        else:
            y_signal = append(y_signal,y(i, z, a1, a2, A_c, A_s, F, BA, 1),0)
            meas_matrix = append(meas_matrix,D(i,z,0,1),0)
            inv_mat = invert_matrix(i,z,inv_mat)
               
            est_tau, est_F, Estimator = estimation_BLUE(inv_mat,meas_matrix,y_signal,1)

            z = innovation_zeta(z,i,est_tau,Estimator,0)
            print variational(i,est_tau,dt*N)
            print z[i]
            estimation_F = append(estimation_F,est_F)
            estimation_tau = append(estimation_tau,est_tau)
    statistics_F = append(statistics_F, est_F)
    statistics_tau = append(statistics_tau, est_tau)
    
#--------------------------------------------------------------------------------
# Print result
#--------------------------------------------------------------------------------

if cycles == 1:
    printout(est_F,est_tau,estimation_F,estimation_tau,z)
else:
    mean_value(statistics_F,statistics_tau)