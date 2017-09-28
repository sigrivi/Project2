import numpy as np
import math
import matplotlib.pyplot as plt
from scipy import linalg
import sys
import time


## THIS IS A SET OF UNIT TESTS FOR THE FUNCTIONS IN PROJECT2B ##

## makes a tridiagonal matrix. A[i,i] = 2/h**2+rho[i+1]**2, A[i,i+1] = -1/h**2, A[i+1,i] = -1/h**2
from project2b import elements_of_A ## Arguments: (maxium rho value, dimension of matrix). Returns: (matrix)

## finds the maximum non-diagonal element of a matrix
from project2b import max_of ## Argument: (matrix). Returns: (absoulute value of largest nondiagonal element, row index, column index)

## rotates matrix around an angle theta
from project2b import rotate_A ## Arguments: (matrix, row index, column index). Returns: (matrix)

##preforms the jacobi iterations
from project2b import jacobi ## Arguments : (matrix, tolerance). Returns: (matrix, number of iterations)




## unit test for max_of
A = np.asarray([[1,2,3,-11,12],[1,4,5,-2,6],[4,7,-1,-1,1],[3,-15,8,-9,1],[-6,-10,-4,2,4]])
answ=(15,3,1) 
assert max_of(A)==answ


## unit test for elements_of_A
E = elements_of_A(10,5)
off_diag = -0.25
for i in range(4):
	assert E[i,i+1] == off_diag
	assert E[i+1,i] == off_diag

diag = np.zeros(5)
for i in range(5):
	diag[i]=0.5+4*(i+1)**2
for i in range(4):
	assert E[i,i] == diag[i]

## unit test for rotate_A
B = rotate_A(E,4,3)
assert B[3,4] == 0
t = -72 + math.sqrt(72**2+1)
assert B[3,3] == (64.5+.5*t+100.5*t**2)/(t**2+1)

F = jacobi(E,1.e-10)[0]
#print(F)
#print(linalg.eig(E)[0])

