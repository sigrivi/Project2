import numpy as np
import math
import matplotlib.pyplot as plt
from scipy import linalg
import sys
import time


## makes a tridiagonal matrix. A[i,i] = 2/h**2+rho[i+1]**2, A[i,i+1] = -1/h**2, A[i+1,i] = -1/h**2
from project2b import elements_of_A ## Arguments: (maxium rho value, dimension of matrix). Returns: (matrix)


## finds the maximum non-diagonal element of a matrix
from project2b_alt import max_of ## Argument: (matrix). Returns: (absoulute value of largest nondiagonal element, row index, column index)

## uses jacobi's algorithm to find eigenvalues and eigenvectors of matrix
from project2b_alt import eigenvalues_and_eigenvectors ## Arguments: (matrix, tolerance epsilon). Returns: (vector of eigenvalues, matrix of eigenvectors). both eigenvalues and eigenvectors are sorted. 


## unit test for max_of
A = np.asarray([[1,2,3,-11,12],[1,4,5,-2,6],[4,7,-1,-1,1],[3,-15,8,-9,1],[-6,-10,-4,2,4]])
answ=(15,3,1) 
assert max_of(A)==answ

E = elements_of_A(10,5)
eig_val, eig_vec = eigenvalues_and_eigenvectors(E, 1.e-8)

## test to check orthonormaility
assert np.dot(eig_vec[0],eig_vec[1])<1.e-10
assert np.abs(np.dot(eig_vec[3],eig_vec[3])-1.)  <1.e-10


## test to check eigenvalues
a,Z = linalg.eigh(E)
print(np.abs(a-eig_val).sum())
assert np.abs(a-eig_val).sum()< 1.e-10
