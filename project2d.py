import numpy as np
import math
import matplotlib.pyplot as plt
from scipy import linalg
import sys
import time


##preforms the jacobi iterations
from project2b import jacobi ## Arguments : (matrix, tolerance). Returns: (matrix, number of iterations)


def elements_of_A(rho_N, N, omega): #makes a tridiagonalmatrix A. Arguments: maximal rho value(rho_max), dimension of matrix(N), and dimensionless frequency (omega)
	A = np.zeros((N,N))
	h = rho_N/N #rho_0=0
	rho = np.zeros(N+1)
	V = np.zeros(N+1)
	for i in range(1,N+1):
		rho[i] = i*h
		V[i] = ( rho[i]**2 )*(omega**2) + 1/rho[i]
	for i in range(N):
		A[i,i] = 2/h**2+V[i+1]
	for i in range(N-1):
		A[i,i+1] = -1/h**2
		A[i+1,i] = -1/h**2
	return(A)

A = elements_of_A(15,100,1/55)

B, iterations = jacobi(A,1.e-8)

eigenvalues = B.diagonal()
eigenvalues = np.sort(eigenvalues)
print(eigenvalues)


