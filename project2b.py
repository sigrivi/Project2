import numpy as np
import math
import matplotlib.pyplot as plt
from scipy import linalg
import sys
import time


## functions

def elements_of_A(rho_N, N): #makes a tridiagonalmatrix A. Arguments: maximal rho value(rho_max) and dimension of matrix(N)
	A = np.zeros((N,N))
	h = rho_N/N #rho_0=0
	rho = np.zeros(N+1)
	V = np.zeros(N+1)
	for i in range(N+1):
		rho[i] = i*h
		V[i] = rho[i]**2
	for i in range(N):
		A[i,i] = 2/h**2+V[i+1]
	for i in range(N-1):
		A[i,i+1] = -1/h**2
		A[i+1,i] = -1/h**2
	return(A)


def rotate_A(A, k, l): #rotates matrix A around an angle theta. The operations are on matrix elements A_kk, A_kl, A_lk, A_ll
	
	N = A.shape[0]
	B = A.copy()
	tau = ( A[k,k]-A[l,l] )/( 2*A[k,l] )
	if A[k,k]-A[l,l] == 0:
		print(" A[k,k]-A[l,l] = 0")
	if A[k,l] ==0:
		print("A[kl}=0")

	t1 = tau - math.sqrt( tau**2+1 ) # t=tan(theta)
	t2 = tau + math.sqrt( tau**2+1 )
	t = t1
	if t2**2 < t1**2: #choose the smaller value of t1 and t2
		t = t2
	for i in range(N):
		if (i!=l and i!=k):
			B[i,k] = (A[i,k]-A[i,l]*t)*(t**2+1)**(-0.5)
			B[i,l] = (A[i,l]+A[i,k]*t)*(t**2+1)**(-0.5)
			B[k,i] = B[i,k]
			B[l,i] = B[i,l]

	B[l,l] = (A[l,l] + 2*A[k,l]*t + A[k,k]*t**2)/(t**2+1)
	B[k,k] = (A[k,k] - 2*A[k,l]*t + A[l,l]*t**2)/(t**2+1)
	B[k,l] = 0
	B[l,k] = 0

	return(B)



def max_of(A): #finds the maximum non-diagonal value
	C = np.absolute(A)
	for i in range(C.shape[0]):
		C[i,i] = 0
	max = np.argmax(C)
	maxvalue = np.max(C)
	m = max//(C.shape[0])
	n = max%(C.shape[0])
	return(maxvalue, m,n) #m is the row index of the maximum element, n is the column index

def jacobi(A, epsilon): #performs the jacobi iterations. Arguments: matrix A and tolerance epsilon
	maxvalue, m,n = max_of(A)
	n_iter=0
	x = np.ones(A.shape[0])
	while (maxvalue>epsilon):
		A = rotate_A(A,m,n)
		maxvalue, m,n = max_of(A)
		n_iter+=1
		#print(n_iter,maxvalue)
	return(A,n_iter)		
	
def eigenvalues(A):
	eigenvalue = np.zeros(A.shape[0])
	for i in range(A.shape[0]):
		eigenvalue[i] = A[i,i]
	eigenvalue.sort()
	return(eigenvalue)
		
A = elements_of_A(10,100) ## N = 150 makes the three first eigenvalues converge with four leading digits

B, iterations = jacobi(A,1.e-8)

print(np.sort(np.real(linalg.eigh(A)[0]))) #to compare eigenvalues with eigenvalues from python solver

X = linalg.eigh(A)[1]
x1 = X[0]**2
x2 = X[1]**2
plt.plot(x1)
plt.show()
#print(eigenvalues(B))





