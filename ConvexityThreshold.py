###################################################################################
##################### Computation of the Convexity Threshold ######################
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
#   For theoretical aspects from which the algorithm is built, see [1] :
#   "Eventual convexity of probability constraints with elliptical distributions" 
#   from Wim Van Ackooij and Jerome Malick
#
#   This file is aimed computing the threshold p* such that the field M(p) := {x, phi(x)>p}
#   is convex for all p>p*. The method "compute" computes the threshold p*. 
#

#from InputsQuadraticExample import *
from InputsSecondQuadraticExample import *
from scipy import integrate
from scipy import special as sp
import math

class ConvexityThreshold :
    
    def __init__(self):
        
        self.inputs = InputClass()
        
        self.precision_on_q = 0.0001
        
        self.precision_on_delta = 0.000001
        
        self.step = 0.001
        
        self.start_evaluation = 0.001
        
    def delta(self,q): # following advice from [1], lemma 2.3 we use dichotomy to find delta(q) 
        a = 0.0 
        b = 1.0
        eval_a = self.equation(a,q)
        eval_b = self.equation(b,q)
        m = (a+b) / 2.0
        eval_m = self.equation(m,q)
        while (abs(eval_m) > self.precision_on_delta) :
            if (eval_m * self.equation(a,q) > 0.0) :
                a = m
            else :
                b = m
            m = (a+b) / 2.0
            eval_m = self.equation(m,q)
        return m
    
    def threshold(self,q):
        delta_q = self.delta(q)
        deltaNd = self.inputs.deltaNd
        t_star = self.inputs.alpha_concavity_threshold
        res = 0
        if (self.inputs.finite_case):
            res = 2 * q + 2*(0.5 - q) * self.inputs.Fr( t_star * self.inputs.bound_M * deltaNd / delta_q )
        else:
            res = (0.5 - q) * self.inputs.Fr( t_star * deltaNd / delta_q ) + (0.5) + q
        return res
    
    def compute(self):
        print("Optimal threshold being computed \n")
        q = self.start_evaluation
        threshold = 2.0 
        best_threshold = threshold
        counter = 0
        print "Result printed when counter reaches : " + str(0.5/self.precision_on_q)
        while (q < 0.5 - self.precision_on_q):
            threshold = self.threshold(q)
            if (best_threshold > threshold):
                best_threshold = threshold
            q += self.precision_on_q
            
            if counter%1000 == 0:
                print(counter)
            counter += 1
        return best_threshold 

    ###################################################################
    ###The following methods are useful for the computation of delta###
    ###################################################################
    
    def regularized_incomplete_beta(self,a,b,c): # the function itself is not defined in the scipy module
        
        res = sp.betainc(a,b,c)
        return res

    def equation(self, delta, q): 
        
        res = self.regularized_incomplete_beta((self.inputs.m-1)/2.0, 0.5, math.sin(math.acos(delta))**2) 
        res += -1.0 + 2*q
        return res;

seuil = ConvexityThreshold()
print(seuil.compute())    
    

