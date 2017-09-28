import numpy as np
import math
import matplotlib.pyplot as plt
from scipy import linalg
import scipy as sp
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


def rotate_A(A, X, k, l): #rotates matrix A around an angle theta. 
	
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
	
	c = (t**2+1)**(-0.5) ## c = cos(theta)
	s = c*t ## s = sin(theta)
	
	Y = X.copy()

	for i in range(N):
		if (i!=l and i!=k):
			B[i,k] = (A[i,k]-A[i,l]*t)*(t**2+1)**(-0.5)
			B[i,l] = (A[i,l]+A[i,k]*t)*(t**2+1)**(-0.5)
			B[k,i] = B[i,k]
			B[l,i] = B[i,l]

		Y[i,k] = c*X[i,k] - s*X[i,l]
		Y[i,l] = c*X[i,l] + s*X[i,k]

	B[l,l] = (A[l,l] + 2*A[k,l]*t + A[k,k]*t**2)/(t**2+1)
	B[k,k] = (A[k,k] - 2*A[k,l]*t + A[l,l]*t**2)/(t**2+1)
	B[k,l] = 0
	B[l,k] = 0

	return(B,Y)



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
	X = np.identity(A.shape[0])
	while (maxvalue>epsilon):
		A,X = rotate_A(A,X,m,n)
		maxvalue, m,n = max_of(A)
		n_iter+=1
		#print(n_iter,maxvalue)
	return(A, X, n_iter)		
	

		
A = elements_of_A(10,60) ## N = 150 makes the three first eigenvalues converge with four leading digits
C = np.copy(A)
B, X, iterations = jacobi(C,1.e-8) #B is the matrix with eigenvalues along diagonal, X is the matrix of eigenvectors

eigenvalues = B.diagonal() #eigenvalues is a vector of eigenvalues
#print(eigenvalues)
eig_sort = np.sort( eigenvalues )
index = np.argsort (eigenvalues )
X=X.transpose() #the eigenvectors are now rows in X
eigenvector = np.zeros((B.shape[0],B.shape[1]))
for i in range(B.shape[0]):
	eigenvector[i,:] = X[index[i],:]
	

#print(eig_sort)
#print(index)

#print(np.sort(np.real(linalg.eigh(A)[0]))) #to compare eigenvalues with eigenvalues from python solver
D = elements_of_A(10,60)
a,Z = linalg.eigh(D)
#a = np.real(linalg.eigh(A)[0])
print(a)

z0 = Z[0]**2
z1 = Z[1]**2

innerproduct = np.dot( eigenvector[3], eigenvector[3])
#print(innerproduct)

#print(np.abs(X-Z))
#plt.plot(X[2,:]**2)
plt.plot(eigenvector[1]**2)
plt.plot(z1)
plt.show()
#print(eigenvalues(B))





