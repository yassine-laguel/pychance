###################################################################################
############################# Data Plotting Class   ###############################
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
#   This class is aimed at plotting the values of phi.
# 

from MonteCarlo import * 
from OptimizedMonteCarlo import *
from DataStorage import *
import matplotlib.pyplot as plt

class Plotter :
    
    def __init__(self, x_min = -2.0 ,x_max = 5.0, y_min = -3.0, y_max = 4.0, method = 0, nb_points = 10, nb_iterations = 100):
        
        self.phi_values = DataStorer(x_min = x_min ,x_max = x_max, y_min = y_min, y_max = y_max , method = method, nb_points = nb_points,  nb_iterations = nb_iterations)
        self.phi_values.compute_phi_table()   
        #self.phi_values.read_phi_table_from_file()
    
    def plot_phi_heatmap(self):
        fig, ax = plt.subplots()
        im = ax.pcolormesh(self.phi_values.X,self.phi_values.Y,self.phi_values.Z)
        fig.colorbar(im)
        ax.axis('tight')
        plt.show()
        
    def plot_phi_heatmap_with_contour_lines(self):
        plt.pcolormesh(self.phi_values.Z)
        plt.colorbar()

        levels = np.linspace(0.1,0.8,15)
        #levels = [0.5,0.977287119533]
        plt.contour(self.phi_values.Z, colors="red", levels=levels)

        fig = plt.gcf()
        fig.set_size_inches(18.5, 10.5, forward=True)
        fig.savefig('test-second-example-quadratique.png', dpi=100)
        plt.show()

# You can test this class with the following commands          
        
print("Phi's heatmap by classical Monte-Carlo method")
graph = Plotter(method = 0, nb_iterations = 50, nb_points = 30)
graph.plot_phi_heatmap_with_contour_lines()

print("Phi's heatmap by antithetic Monte-Carlo method")
graph = Plotter(method = 1, nb_iterations = 50, nb_points = 30)
graph.plot_phi_heatmap_with_contour_lines()

print("Phi's heatmap by optimized Monte-Carlo method")
graph = Plotter(method = 2, nb_iterations = 50, nb_points = 30)
graph.plot_phi_heatmap_with_contour_lines()
