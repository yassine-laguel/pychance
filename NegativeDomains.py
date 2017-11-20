###################################################################################
################### Negative Domains of the function t -> g(x,mu+tLv) #############
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
#   We compute here the domains where the function t -> g(x,mu+tLv)
#   is negative. The algorithm depends on the informations the user 
#   gives in the file Inputs.py.
# 
#   The problem and related techniques that appear here do not depend on the 
#   method that will be used to compute the probability IP( g(x,\xi) <= 0 ) 
#   (which can be Monte Carlo or Quasi Monte Carlo for instance)
#   It rather depends on the information on the informations given on g, like its 
#   derivative when given.
#
#   The main method of this class is "domains" which returns a list of couples [a_i,b_i]
#   such that a_0 < b_0 < a_1 < ... < a_n < b_n  and for which g( x , mu + tLv ) < 0 
#   for any t in [a_i,b_i] .
#

from scipy import optimize as optim
import numpy as np
import sys

sys.setrecursionlimit(100000)  # useful for the recursive calls of the function interval_with_root

# python Class aimed at storing the domains where the function t -> g(x,mu+tLv) is negative. 
class zeros_of_g:
    def __init__(self, inputs, x, v):
        
        self.inputs = inputs # information given by the user in the file Inputs.py 
        
        self.x = x  # value of x used in the computation g(x,mu+tLv) 
                    # Dimension of x must be inputs.n
            
        self.Lv = np.dot(inputs.L, v) # value of the vector Lv used in the computation 
                    # of g(x,mu+tLv). L is the Choleski matrix such that R = L^tL
            
        self.precision = 0.001  # We suppose here that for any real numbers a < b such that b-a < self.precision
                                # there is no t real strictly in [a,b] such that g(x,mu+tLv) == 0
        
        
        self.borne = 20.0 # We don't study the domains that are not in [0,self.borne]
                           # Indeed as we compose Fr by rho and as Fr decreases very quickly, 
                           # we do here the approximation Fr(t) = 0 if t > 100
                
        self.free_roots_domains = [[self.borne,
                                    self.borne - self.precision]]  ### we suppose here that there aren't roots at the extremity of the given interval

    ### Returns g(x,mu+tLv)
    def g_xv(self, t):
        res = self.inputs.g(self.x, self.inputs.mu + t * self.Lv)
        return res

    ### Returns the gradient of g_xv evaluated in t
    def gradient_g_xv(self, t):
        A = self.inputs.g_jac_z(self.x, self.inputs.mu + t * self.Lv)
        res = np.dot(A, self.Lv)
        return res[0,0]

    ### Returns a zero using Newton-Raphson's method
    def nearest_root_by_newton(self, t):
        return optim.newton(self.g_xv, t, self.gradient_g_xv)

    ### Returns a root of g_xv in [a,b] using a mixed Newton's algorithm
    ### and secant method. Convergence is guarented.
    ### Needs f(a)*f(b) < 0
    def root_in(self, a, b):
        u = a
        v = b
        res = v - self.g_xv(v) * (v - u) / (self.g_xv(v) - self.g_xv(u))
        while (abs(self.g_xv(res)) > 0.00000001):

            if (-self.g_xv(res) * self.g_xv(a) > 0):
                v = res
            else:
                u = res

            if (self.gradient_g_xv(v) != 0):
                res = v - self.g_xv(v) / self.gradient_g_xv(v)
            if not ((res >= u) and (res <= v)):
                res = v - self.g_xv(v) * (v - u) / (self.g_xv(v) - self.g_xv(u))

        return res

    ### Returns an interval in [a,b] which contains a root of g_xv if this root exists
    ### else returns [-1,-1]. Algorithm based on bisection.
    def interval_with_root(self, a, b):
        #print("Use of Interval_with_roots")


        if (a == -1):
            return [-1, -1]
        elif (-1 * self.g_xv(a) * self.g_xv(b) > 0):  # if a et b produce opposite signs
            return [a, b]

        elif (b - a < self.precision):  # we suppose here that two roots cannot be separated by a distance less than self.precision
             return [-1, -1]

        else:
            m = (a + b) / 2.0
            res0 = self.interval_with_root(a, m)
            if (res0[0] != -1):
               return res0
            else:
               return self.interval_with_root(m, b)

    ### Returns a list of the roots of g_xv in the interval [a,b]
    def list_of_roots(self, a, b):
        interval = self.interval_with_root(a, b)
        if (interval[0] == -1):
            return [];
        else:
            x = self.root_in(interval[0], interval[1])
            res = self.list_of_roots(a, x - self.precision)
            res.append(x)
            list_aux = self.list_of_roots(x + self.precision, b)
            res.extend(list_aux)
            return res

    ### Returns the list of intervals in which g_xv is negative
    def domains_general_case(self):
        
        l = [0.0]
        roots = self.list_of_roots(0, self.borne)
        l.extend(roots)
        l.append(self.borne)
        n = len(l)
        counter = 0
        res = []

        while counter + 1 < n:
            x = self.g_xv( (l[counter] + l[counter + 1]) / 2)
            if x < 0:
                res.append([l[counter], l[counter + 1]])

            counter += 1
            
        return res
    
    def domains_in_convex_case(self):
        if self.g_xv(self.borne) > 0.00000001 :
            rho = self.root_in(0.0,self.borne) # a direct newton_raphson would be convenient 
                                               # but the one from scipy.optimize doesn't work because
                                               # (it seems to change modyfy and recall the estimation of the root you
                                               # give to it...)
            return [[0.0,rho]]
        else :
            return [[0.0,self.borne]]
        
    def domains(self):
        if self.inputs.g_convex_in_z :
            return self.domains_in_convex_case()
        else :
            return self.domains_general_case()
