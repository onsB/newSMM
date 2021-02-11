# -*- coding: utf-8 -*-
"""
Created on Mon Oct 26 23:18:33 2020

@author: onsbo
"""

import matplotlib.pyplot as plt




def newton(function, derivative, x0, tolerance, number_of_max_iterations=100):
    x1 = 0

    if abs(x0-x1)<= tolerance and abs((x0-x1)/x0)<= tolerance:
        return x0

    print("k\t x0\t\t function(x0)")
    k = 1

    while k <= number_of_max_iterations:
        x1 = x0 - (function(x0)/derivative(x0))
        #print("x%d\t%e\t%e"%(k,x1,function(x1)))

        if abs(x0-x1)<= tolerance and abs((x0-x1)/x0)<= tolerance:
            plt.plot(x0, function(x0), 'or')
            return x1

        x0 = x1
        k = k + 1
        plt.plot(x0, function(x0), 'or')

        # Stops the method if it hits the number of maximum iterations
        if k > number_of_max_iterations:
            print("newton-raphson Exceeded max number of iterations")

    return(x1) # Returns the value

