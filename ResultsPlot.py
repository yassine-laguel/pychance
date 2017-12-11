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
import matplotlib.pyplot as plt
from chance_utils import *

from MonteCarlo import *
from OptimizedMonteCarlo import *
from ConvexityThreshold import *

from DataStorage import *


class Plotter :
    
    def __init__(self, x_min = -20.0 ,x_max = 20.0, y_min = -20.0, y_max = 20.0, method = 2, nb_points = 30, nb_iterations = 20):

        # class containing specified functions of the problem (launched from the file Inputs.py)
        # self.inputs = InputClass()

        # Values of phi computed
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
        plt.pcolormesh(self.phi_values.X, self.phi_values.Y, self.phi_values.Z)
        plt.colorbar()

        levels = np.linspace(0.1, 0.95, 15)
        #levels = [0.5,0.977287119533]
        plt.contour(self.phi_values.X, self.phi_values.Y, self.phi_values.Z, colors="red", levels=levels)

        fig = plt.gcf()
        fig.set_size_inches(18.5, 10.5, forward=True)
        fig.savefig('test-second-example-quadratique.png', dpi=100)
        plt.show()


    ####################################################################################################################
    ############################################  Proposed  Methods  ###################################################
    ####################################################################################################################

    def plot_rho_levelset(self, thresh):

        convexity_thresholder = ConvexityThreshold()

        fig, ax = plt.subplots()
        im = ax.pcolormesh(self.phi_values.X, self.phi_values.Y, self.phi_values.Z)
        for x in self.phi_values.X:
            for y in self.phi_values.Y:
                u = make_vector(x, y)
                if convexity_thresholder.above_threshold(u, thresh):
                    ax.scatter(x, y, marker=',', color='red')
        fig.colorbar(im)
        figure = plt.gcf()
        figure.set_size_inches(18.5, 10.5)
        ax.axis('tight')
        plt.show()

    def plot_rho_levelset_directed(self, thresh, direction):

        convexity_thresholder = ConvexityThreshold()

        fig, ax = plt.subplots()
        im = ax.pcolormesh(self.phi_values.X, self.phi_values.Y, self.phi_values.Z)
        for x in self.phi_values.X:
            for y in self.phi_values.Y:
                u = make_vector(x, y)
                if convexity_thresholder.above_directed_threshold(u, direction, thresh):
                    ax.scatter(x, y, color='red')
        fig.colorbar(im)
        figure = plt.gcf()
        figure.set_size_inches(18.5, 10.5)
        ax.axis('tight')
        plt.show()

    def plot_rho_levelset_with_contour_lines(self, thresh):

        convexity_thresholder = ConvexityThreshold()

        fig, ax = plt.subplots()
        im = ax.pcolormesh(self.phi_values.X, self.phi_values.Y, self.phi_values.Z)
        levels = np.linspace(0.1, 0.95, 15)
        ax.contour(self.phi_values.X, self.phi_values.Y, self.phi_values.Z, colors="red", levels=levels)
        for x in self.phi_values.X:
            for y in self.phi_values.Y:
                u = make_vector(x, y)
                if convexity_thresholder.above_threshold(u, thresh):
                    ax.scatter(x, y, marker=',', color='red')
        fig.colorbar(im)
        figure = plt.gcf()
        figure.set_size_inches(18.5, 10.5)
        ax.axis('tight')
        plt.show()

    def plot_convex_set_obtained_from_threshold(self):

        threshold_computer = ConvexityThreshold()
        threshold = threshold_computer.compute()
        fig, ax = plt.subplots()
        im = ax.pcolormesh(self.phi_values.X, self.phi_values.Y, self.phi_values.Z)
        levels = np.linspace(0.1, 0.95, 15)
        ax.contour(self.phi_values.X, self.phi_values.Y, self.phi_values.Z, colors="red", levels=levels)
        x_size = len(self.phi_values.X)
        y_size = len(self.phi_values.Y)
        for x_index in range(x_size):
            for y_index in range(y_size):
                if self.phi_values.Z[x_index, y_index] > threshold - 0.1:
                    ax.scatter(self.phi_values.X[x_index], self.phi_values.Y[y_index], marker=',', color='red')
        fig.colorbar(im)
        figure = plt.gcf()
        figure.set_size_inches(18.5, 10.5)
        ax.axis('tight')
        plt.show()


# You can test this class with the following commands          
        
#print("Phi's heatmap by classical Monte-Carlo method")
#graph = Plotter(method = 0, nb_iterations = 50, nb_points = 30)
#graph.plot_phi_heatmap_with_contour_lines()

#print("Phi's heatmap by antithetic Monte-Carlo method")
#graph = Plotter(method = 1, nb_iterations = 50, nb_points = 30)
#graph.plot_phi_heatmap_with_contour_lines()

#print("Phi's heatmap by optimized Monte-Carlo method")
#graph = Plotter(method = 2, nb_iterations = 20, nb_points = 30)
#graph.plot_phi_heatmap_with_contour_lines()
