##############################################################################
## Algorithms for the study of feasible sets of Chance Constrained Problems ##
##############################################################################

Author Yassine Laguel
Last Modification : 11/08/2017

We consider a Chance Constraint Problem of the form
phi(x) = IP( g(x,\xi) <= 0 ),

xi is an elliptical random vector of dimension m, with 
	mean mu and correlation matrix R,

x is assumed to be such that g(x,mu) < 0

For theoretical aspects from which the algorithm is built, see [1] :
"Eventual convexity of probability constraints with elliptical distributions" 
from Wim Van Ackooij and Jerome Malick

This projects is aimed at computing and plotting the function phi as well as computing
the threshold p* such that the levelset {x, phi(x)> p} is convex for any p > p*

To implement a given problem, the user has to fill in the file Inputs.py
	The file InputsQuadraticExample.py can be seen as an example

The files DataStorage.py and ResultsPlots.py are respectively aimed at 
storing the values of phi and plotting them. 

The file NegativeDomains.py, MonteCarlo, AntitheticMonteCarlo and 
OptimizedMonteCarlo are aimed at computating phi'svalue. 
	Warning : If you want to solve an example of problem, don't forget to import the InputFile in 
  		  NegativeDomains.py, MonteCarlo, AntitheticMonteCarlo and 
		  OptimizedMonteCarlo

The file Threshold.py is aimed at computing the value p^*


