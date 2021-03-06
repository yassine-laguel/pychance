###################################################################################
################### Monte Carlo method to compute IP( g(x,xi) < 0 ) ###############
###################################################################################

#
#   Author Yassine Laguel
#   Last Modification : 10/08/2017
#
#   Algorithm based on the theoretical paper :
#   "Eventual convexity of probability constraints with elliptical distributions" 
#   from Wim Van Ackooij and Jerome Malick 
#
#   We consider a Chance Constraint Problem of the form
#   phi(x) = IP( g(x,\xi) <= 0 ),
#
#   xi is an elliptical random vector of dimension m, with 
#      mean mu and correlation matrix R,
#   
#   x is assumed to be such that g(x,mu) < 0
#
#   We use here the Monte_Carlo method to compute IP( g(x,xi) < 0 ) 
#   For the complete presentation of the problem, see the file Inputs.py 
#
#   According to the article "Eventual convexity of probability constraints
#   with elliptical distributions" from Wim Van Ackooij and Jerome Malick
#   we use the radial decomposition of xi : xi = mu + RLV to compute 
#   IP( g(x,xi) < 0 ).
#
#   The elements of the problem the user wants to solve must be completed in the 
#   file Inputs.py (specification of the function g, the function rho if computed,
#   etc...).
#   
#   As an example, one might look at the file InputsQuadraticExample to see 
#   an example of input file
#

#from Inputs import *
# Comment the previous line and uncomment the following one
# if you want to test the quadratic example
from chance_utils import  *
from InputsQuadraticExample import * 
import numpy as np
from NegativeDomains import *


class compute_phi_by_monte_carlo :
    
    def __init__(self, nb_iterations):
        ### class containing specified functions of the problem (launched from the file Inputs.py)
        self.inputs = InputClass()

        ### Number of iterations of the Monte-Carlo method
        self.nb_iterations = nb_iterations


    ### Returns sum(Fr(b_i) - Fr(a_i)) where [a_i,b_i] are domains where t->g(x+tLv) is non-positive
    def ray(self,x,v): 
    
        S = 0
        zeros = zeros_of_g(self.inputs,x,v)
        intervals = zeros.domains()
        n = len(intervals)
    
        for i in range(n):
            S = S + self.inputs.Fr((intervals[i])[1]) - self.inputs.Fr((intervals[i])[0])
        return S

    ### Returns an approximation of phi(x) = Proba(g(x,xi)<0) with Monte-Carlo method.
    def phi(self,x):
        res = 0 
    
        for i in range(self.nb_iterations):
            v = random_v(self.inputs.m)
            v = np.matrix(v)
            if self.inputs.rho_given :
                res += self.inputs.Fr(self.inputs.rho(x,v))
            else :
                res += self.ray(x,v)
        res = res / self.nb_iterations
        return res
