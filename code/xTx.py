# -*- coding: utf-8 -*-
"""
Created on Sun Nov 29 23:47:02 2020

@author: onsbo
"""
import numpy as np

def mult(x):
    return(x.T*x)

def multnp(x):
    return(np.matmul(np.transpose(x),x))
    
def covmult(x,p):
    cov = np.dot(p * (x).T, x)
    return(cov)