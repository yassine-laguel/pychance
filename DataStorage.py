###################################################################################
####################### Data Storage Class of Phi Values   ########################
###################################################################################

#
#   Author Yassine Laguel
#   Last Modification : 10/08/2017
#
#   Algorithm based on the theoretical paper :
#   "Eventual convexity of probability constraints with elliptical distributions" 
#   from Wim Van Ackooij and Jerome Malick 
#
#
#   We consider a Chance Constraint Problem of the form
#   phi(x) = IP( g(x,\xi) <= 0 ),
#
#   xi is an elliptical random vector of dimension m, with 
#      mean mu and correlation matrix R,
#   
#   x is assumed to be such that g(x,mu) < 0
#
#   This class is aimed at storing the values of phi.
#   


from MonteCarlo import * 
from OptimizedMonteCarlo import *
from AntitheticMonteCarlo import *
import json 


class DataStorer :
    def __init__(self, filename = "No_file_for_now", x_min = -100.0 ,x_max = 100.0, y_min = -100.0, y_max = 100.0, method = 0, nb_points = 10, nb_iterations = 100):
        
        self.filename = filename
        self.nb_points = nb_points
        self.x_min = x_min
        self.x_max = x_max
        self.y_min = y_min
        self.y_max = y_max
        self.method = method
        self.nb_iterations = nb_iterations
        self.X = []
        self.Y = []
        self.Z = []
        
    def proba_method(self):
        if self.method == 0: 
            return compute_phi_by_monte_carlo(self.nb_iterations)
        elif self.method == 1:
            return compute_phi_by_antithetic_monte_carlo(self.nb_iterations)
        elif self.method == 2:
            return compute_phi_by_optimized_monte_carlo(self.nb_iterations)
     
    
    ### Returns a double array of values phi([x,y]) for x_min < x < x_max and 
    ### y_min < y < y_max. x and y take each nb_points values in their interval
    def compute_phi_table(self):
        
        phi_calculator = self.proba_method()
        
        X = np.linspace(self.x_min,self.x_max,self.nb_points)
        Y = np.linspace(self.y_min,self.y_max,self.nb_points)
        
        nb_iterations_total = self.nb_points**2

        Z = []
        counter = 0.0
        for y in Y:
            for x in X:
                if counter % 100 == 0 : 
                    print(counter/(nb_iterations_total))
                u = [x,y]
                u = np.matrix(u)
                u = np.transpose(u)
                Z.append(phi_calculator.phi(u))
                counter +=1

        Z = np.reshape(Z, [len(X), len(Y)])
        self.X = X
        self.Y = Y
        self.Z = Z
    
    def write_phi_table_into_file(self):
        valeurs = {}
        counter = 0
        for i in range(self.nb_points):
            for j in range(self.nb_points):
                valeurs[str(counter)] = [self.X[i],self.Y[j],self.Z[i,j]]
                counter +=1
        
        out_file = open("valeurs_phi.json","w")

        # Save the dictionary into this file
        # (the 'indent=4' is optional, but makes it more readable)
        json.dump(valeurs,out_file, indent=4)                                    

        # Close the file
        out_file.close()  
    
    def read_phi_table_from_file(self):
        in_file = open("valeurs_phi.json","r")
        valeurs = json.load(in_file)
        X = []
        Y = []
        Z = []
        counter = 0
        for i in range(self.nb_points):
            for j in range(self.nb_points):
                data_value =  valeurs[str(counter)]
                if i == 0 :
                    Y.append(data_value[1])
                if j == 0 :
                    X.append(data_value[0])
                Z.append(data_value[2])
                counter += 1
        Z = np.reshape(Z, [len(X), len(Y)])
        self.X = X
        self.Y = Y
        self.Z = Z
         