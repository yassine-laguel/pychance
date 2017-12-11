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
from chance_utils import *

from InputsQuadraticExample import *
# from InputsWimQuadraticExample import *
# from scipy import integrate
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

    # return true if we find rho(x,v)> t for a lot of direction v tested
    def above_threshold(self, x, t, nb_iterations=1, rotation_order=20):
        res = True
        rotation = generate_n_order_rotation(self.inputs.m, n=rotation_order)
        for i in range(nb_iterations):
            v = random_v(self.inputs.m)
            rotation_power = rotation.copy()
            for j in range(rotation_order):
                v2 = v.copy()
                if j > 0:
                    rotation_power = np.dot(rotation_power, rotation)
                    v = np.dot(rotation_power, v2)
                if self.inputs.rho_given:
                    # print(self.inputs.rho(x,v))
                    res = res and self.inputs.rho(x, v) >= t
                else:
                    print("Error, rho is not given, we cant plot the designated level-set of rho")

        return res

    # return true if we find rho(x,v)> t for a lot of direction v tested such that <v|direction> is positive
    def above_directed_threshold(self, x, direction, t, nb_iterations=1, rotation_order=20):
        res = True
        for i in range(nb_iterations):
            v = random_v(self.inputs.m)
            rotation = generate_n_order_rotation(self.inputs.m, n=rotation_order)
            rotation_power = rotation.copy()
            for j in range(rotation_order):
                v2 = v.copy()
                if j > 0:
                    rotation_power = np.dot(rotation_power, rotation)
                    v2 = np.dot(rotation_power, v2)
                if self.inputs.rho_given:
                    if np.dot(np.transpose(v2), direction)[0, 0] > 0:
                        # print(self.inputs.rho(x,v))
                        res = res and self.inputs.rho(x, v) >= t
                else:
                    print("Error, rho is not given, we cant plot the designated level-set of rho")

        return res




    ###################################################################
    ###The following methods are useful for the computation of delta###
    ###################################################################
    
    def regularized_incomplete_beta(self,a,b,c): # the function itself is not defined in the scipy module
        
        res = sp.betainc(a,b,c)
        return res

    def equation(self, delta, q): 
        
        res = self.regularized_incomplete_beta((self.inputs.m-1)/2.0, 0.5, math.sin(math.acos(delta))**2) 
        res += -1.0 + 2*q
        return res

    

