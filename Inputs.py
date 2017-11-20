###################################################################################
################### Inputs of the Chance Constraint Problem #######################
###################################################################################

#
#   Author Yassine Laguel
#   Last Modification : 11/08/2017
#
#   We consider a Chance Constraint Problem of the form
#   phi(x) = IP( g(x,\xi) <= 0 ),
#
#   xi is an elliptical random vector of dimension m, with 
#      mean mu and correlation matrix R,
#   
#   x is assumed to be such that g(x,mu) < 0
# 
#   For theoretical aspects from which the algorithm is built, see [1] :
#   "Eventual convexity of probability constraints with elliptical distributions" 
#   from Wim Van Ackooij and Jerome Malick
#
#   This file is aimed at receiving all the inputs of the chance constraint problem the 
#   user wants to solve.
#
#   Some informations are not necessary but very helpful to compute faster 
#   and with more precision IP( g(x,\xi) <= 0 )
#
#   As an example, one might look at the file InputsQuadraticExample to see 
#   how this file should be filled in.
#

import numpy as np
import math


class InputClass :
    
    ######################################################################
    ################## Information given by the user #####################
    ######################################################################
    
    def __init__(self):
        
        # Dimension of the first variable of g
        self.n = 2
        
        # Dimension of the second variable of g
        self.m = 2
        
        # Mean of xi
        self.mu = np.transpose(np.matrix([0, 0]))
        
        self.L = np.identity(2) # Choleski Matrix appearing in radial decoposition of xi.
                                # L is such that L^tL = R where r is the covariance matrice of xi
        
        self.rho_given = False  # rho_given must have true value iff the user fills in the rho method
                                # Speeds the resolution
        
        self.g_convex_in_z = False  # g_convex_in_z must be true iff g is convex in its second variable
                                    # Speeds the resolution
            
            
        self.deltaNd = 1 # deltaNd is the real number satisfying rho^co < deltaNd * rho 
                         # where rho^co is the function defined in [1]
            
            
        self.alpha_concavity_threshold = 10 # threshold from which alpha_revealed concavity 
                                            # of Fr and alpha-concavity if rho are guaranted. 
                
            
        self.finite_case = False # finite_case must be true iff the set M(x) = {v, g(x,v)<0} is bounded for any x
        self.bound_M = 10.0 # bound for the function C(x) = sup_{v1,v2 in sphere} \frac[\rho(x,v1)][\rho(x,v2)]
                            # must be given if finite_case == true
                            
        
        print("Ensemble des données chargées !\n")

    ### function g chosen
    def g(self,x,z):
        print("Stop ! g still not defined")
    
    ### Jacobian matrix of g according to x
    def g_jac_x(self,x,z):
        print("Stop ! g_jac_x still not defined")
    
    ### Jacobian matrix of g according to z
    ### If rho is not given, this method must be written in order
    ### to compute IP( g(x,\xi) <= 0 )
    def g_jac_z(self,x,z):
        print("g_jac_z still not defined")
    
    ### Random variable chosen
    def xi(self):
        print("Fr still not defined")
    
    ### Return the mean of xi
    def getMu(self):
        return self.mu
    
    ### Return the Choleski Matrix appearing in radial decoposition of xi
    def getL(self):
        return self.L
    
    ### Return Fr(t) where Fr is the repartition function of the radial distribution
    ### appearing in the radial decomposition of xi
    def Fr(self,t):
        print("Fr still not defined")
    
    ### function rho :
    def rho(self,x,v):
        print("rho still not defined")   
        

