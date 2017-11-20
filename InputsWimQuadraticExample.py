###################################################################################
########################## Example of the quadratic problem #######################
###################################################################################

#
#   Author Yassine Laguel
#   Last Modification : 10/08/2017
#
#   Algorithm based on the theoretical paper :
#   "Eventual convexity of probability constraints with elliptical distributions" 
#   from Wim Van Ackooij and Jerome Malick 
#
#   This file stands as an example of input of chance constraint problem.  
#
#   We consider a Chance Constraint Problem of the form
#   phi(x) = IP( g(x,\xi) <= 0 ),
#
#   xi is an elliptical random vector of dimension m, with 
#      mean mu and correlation matrix R,
#   
#   g is here of the form g(x,z) = <z|W(x)z> + <beta|z> + b
#      where 
#           W(x) = x_1 W1 + x_2 W2, with x_1,x_2 positive and W1,W2 positive semi-definite 
#
#           beta is a constant in R^m
#   
#           b is a real constant satisfying b < 0
#
#

import numpy as np
import math
from scipy.special import gammainc as gamma
from scipy.stats import ortho_group as ortho

class InputClass:
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

        # Choleski Matrix appearing in radial decoposition of xi
        self.L = self.compute_L()
        
        ### We are here able to compute directly rho. In that way, the class Negative_Domains
        ### won't be needed for the computation of IP( g(x,\xi) <= 0 ).
        self.rho_given = True # rho_given must have true value iff the user fills in the rho method
                               # Speeds the resolution
        self.g_convex_in_z = True  # g_convex_in_z must be true iff g is convex in its second variable
                                   # Speeds the resolution
        
        
        self.deltaNd = 1 # deltaNd is a realnumber satisfying rho^co < deltaNd * rho 
                         # where rho^co is the function defined in [1]
                         # Used if self.g_convex_in_z = False
                
        self.alpha_concavity_threshold = math.sqrt(self.m + 3) # threshold from which alpha_revealed concavity 
                                                               # of Fr and alpha-concavity if rho are guaranted. 
        
        
        self.finite_case = False # finite_case must be true iff the set M(x) = {v, g(x,v)<0} is bounded for any x
        self.bound_M = 10.0 # bound for the function C(x) = sup_{v1,v2 in sphere} \frac[\rho(x,v1)][\rho(x,v2)]
                            # must be given if finite_case == true
        
        
        ###################################################################
        ####The following variables are linked to this specific problem####
        ###################################################################

        self.b = -3 # # variable linked to this particular problem added
        
        self.rotation =  ortho.rvs(dim=2)

        print("Ensemble des données chargées !\n")

    ### function g chosen -- unused here since rho is given
    def g(self, x, z):
        a = np.dot(np.transpose(z),np.dot(self.W(x),z))
        b = np.dot(self.beta,z)
        c = self.b
        res = a + b + c
        return res[0,0]

    ### Jacobian matrix of g according to x -- unused here since rho is given
    def g_jac_x(self, x, z):
        print("Stop ! g_jac_x still not defined")
        #return 1

    ### Jacobian matrix of g according to z -- unused here since rho is given
    def g_jac_z(self, x, z):
        a = 2 * np.dot(self.W(x),z)
        res = np.transpose(a) + self.beta
        return res

    ### Random variable chosen -- unused here since rho is given
    def xi(self):
        print("Stop ! xi still not defined")
        #return res

    ### Return the mean of xi
    def getMu(self):
        return self.mu

    ### Return the Choleski Matrix appearing in radial decoposition of xi
    def getL(self):
        return self.L

    ### Return Fr(t) where Fr is the repartition function of the radial distribution
    def Fr(self, t):
        # Test here
        res = gamma(self.m/2,t*t/2)
        return res

    ### function rho chosen:
    def rho(self,x,v):
        Lv = np.dot(self.L,v)
        a = (np.dot(np.transpose(Lv),np.dot(self.W(x),Lv)))[0,0]
        b = self.beta(x,v)
        c = self.b
        res = - b + math.sqrt(b**2 - a*c)
        res = res / a
        return res

    ###################################################################
    #####The following methods are linked to this specific problem#####
    ###################################################################
    def W(self,x):
        W1 = np.matrix('1.0 0.9; 0.9 1.0')
        W2 = np.matrix('1.0 -0.7; -0.7 1.0')
        res = x[0,0] * W1 + x[1,0] * W2
        return res
    
    def beta(self,x,v):
        u = x.copy()
        Lv = np.dot(self.L,v)
        W = np.matrix('-1.0 1.0; 2.0 3.0')        
        res = (np.dot(np.transpose(u),np.dot(W,Lv)))[0,0]
        return res
    
    def compute_L(self):
        c = math.sqrt(1.0 - 0.25)
        res = np.matrix([[1.0,0],[0.5,c]])
        return res
