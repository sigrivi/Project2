import numpy as np
import math
import matplotlib.pyplot as plt
from scipy import linalg
import scipy as sp
import sys
import time


## functions--------------------------------------------------------------------

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
	
	Y = X.copy() #matrix of eigen vectors

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
	return(A, X, n_iter)		
	

def eigenvalues_and_eigenvectors(A, epsilon): # use the jacobi function, then returns sorted eigenvalues as a matrix, and sorted eigenvalues as rows in a matrix
	
	B, X, iterations = jacobi(A,epsilon) #B is the matrix with eigenvalues along diagonal, X is the matrix of eigenvectors
	print("iteration:", iterations)
	eigenvalues = B.diagonal() #eigenvalues is a vector of eigenvalues
	eig_sort = np.sort( eigenvalues )
	index = np.argsort (eigenvalues )
	X=X.transpose() #the eigenvectors are now rows in X
	eigenvector = np.zeros((B.shape[0],B.shape[1]))
	for i in range(B.shape[0]):
		eigenvector[i,:] = X[index[i],:]
	
	return(eig_sort, eigenvector)


if __name__ == '__main__':
	rho_N = 5  ## The maximum rho value
	N = 200 ## N = 200 makes the three first eigenvalues converge with approx. four leading digits after decimal point
	epsilon = 1.e-8 
	rho = np.linspace(0,rho_N,num = N) ## a vector for plotting eigenvectors against rho
	A = elements_of_A(rho_N,N)  ## Make the discretization matrix
	#print(A)

	time1=time.time()
	eig_val, eig_vec = eigenvalues_and_eigenvectors(A, epsilon)  ## Solve the eigenvalue problem
	time2=time.time()
	#print(eig_val)
	#print(eig_vec[0], eig_vec[1], eig_vec[2])
	print(eig_val[0],eig_val[1], eig_val[2]) ## print three lowest eigenvalues
	#print("time jacobi:",(time2-time1))

	plt.plot(rho, eig_vec[0]**2, rho, eig_vec[1]**2, rho, eig_vec[2]**2) ## plot eigenvectors of the three lowest states
	plt.title("Probability distribution for the three lowest energy states")
	plt.xlabel("rho")
	plt.ylabel("radial probability")
	plt.legend(["ground state","1. energy state", "2. energy state"])
	#plt.savefig('one_electron.png',dpi=225)
	plt.show()


	#print(np.sort(np.real(linalg.eigh(A)[0]))) #to compare eigenvalues with eigenvalues from python solver
	D = elements_of_A(rho_N,N)
	time3=time.time()
	a,Z = linalg.eigh(D) ## python eigenvalues solver
	Z=Z.transpose()
	time4=time.time()
	#print("time python",time4-time3)

	plt.plot(rho, Z[0]**2,'r', rho, eig_vec[0]**2, 'b') ## plot python solution together with my solution, to check that they are equal. 
	plt.show()






