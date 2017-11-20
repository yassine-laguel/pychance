###################################################################################
########### Antithetic Monte Carlo method to compute IP( g(x,xi) < 0 ) ############
###################################################################################

#
#   Author Yassine Laguel
#   Last Modification : 11/08/2017
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
#   For the complete presentation of the problem, see the file Inputs.py
#
#   We use here the Monte_Carlo method to compute IP( g(x,xi) < 0 ) 
#   with the difference that for each random spherical vector v generated
#   if v = (v_1,...,v_n), we compute apply Monte_carlo to all the vectors of the form  
#   (+-v_1,...,+-v_n).
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
from InputsQuadraticExample import * 
from NegativeDomains import *
import numpy as np
import math

class compute_phi_by_antithetic_monte_carlo():
    
    def __init__(self, nb_iterations):
        
        # class containing specified functions of the problem (launched from the file Inputs.py)
        self.inputs = InputClass()

        # Number of iterations of the Monte-Carlo method
        self.nb_iterations = nb_iterations 
    
        
    ### Returns a random vector v, uniformly distributed belonging to the sphere. 
    def random_v(self):
        v = np.random.normal(0.0,1,self.inputs.m)
        v = v/np.linalg.norm(v)
        v = np.matrix(v)
        v = np.transpose(v)
        return v

    ### Returns sum(Fr(b_i) - Fr(a_i)) where [a_i,b_i] are domains where t->g(x+tLv) is non-positive
    def ray(self,x,v): 
    
        S = 0
        zeros = zeros_of_g(self.inputs,x,v)
        intervals = zeros.domains()
        n = len(intervals)
    
        for i in range(n):
            S = S + self.inputs.Fr((intervals[i])[1]) - self.inputs.Fr((intervals[i])[0])
        return S
    
    def binary_representation(self,x):
        u = np.binary_repr(x)
        liste = [int(i) for i in str(u)]
        liste.reverse()
        while(len(liste) < self.inputs.m):
            liste.append(0)
        return liste

    def generate_system(self,v,index):
        u = self.binary_representation(index)
        vector = v.copy()
        for i in range(self.inputs.m):
            if u[i] == 1 : 
                vector[i,0] *= -1
        return vector
    
    ### Returns an approximation of phi(x) = Proba(g(x,xi)<0) with an optimized 
    ### version of the Monte-Carlo method.
    def phi(self,x):
        res = 0
        nb_iterations_antithetic = 2**self.inputs.m
        nb_iterations2 = self.nb_iterations / nb_iterations_antithetic
        for i in range(nb_iterations2):
            w = self.random_v()
            for index in range(nb_iterations_antithetic):
                v = self.generate_system(w,index)
                if self.inputs.rho_given :
                    res += self.inputs.Fr(self.inputs.rho(x,v))
                else :
                    res += self.ray(x,v) 
                
        res = res / (self.nb_iterations)
        return res
    




