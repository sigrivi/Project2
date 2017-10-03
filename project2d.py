import numpy as np
import math
import matplotlib.pyplot as plt
from scipy import linalg
import sys
import time


##performs the jacobi iterations
from project2b import jacobi ## Arguments : (matrix, tolerance). Returns: (matrix, number of iterations)

from project2b_alt import eigenvalues_and_eigenvectors ## Arguments (matrix, tolerance) Returns:(eigevalues, eigenvectors)


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
	

def eigenvector_i(rho_N,N,i): ## gives eigenvector x_i for different values of omega. 
	omega = [0.01, 0.5, 1, 5] ## here you can choose different omegas
	X = np.zeros((len(omega),N))
	for j in range(len(omega)):
		A = elements_of_A(rho_N,N,omega[j])
		eig_val, eig_vec = eigenvalues_and_eigenvectors(A,1.e-8)
		X[j] = eig_vec[i] 
	return X
		
	

rho_N = 15
N = 100
#A = elements_of_A(rho_N,N,0.05) #calculate eigenvalues for a given omega
#eig_val, eig_vec = eigenvalues_and_eigenvectors(A,1.e-8)
#print(eig_val)



X = eigenvector_i(rho_N,N,0)


h_bar = 6.582119514*(1.e-16) # in [eVs/rad]
mass = 9.10938356*(1.e-31) # in [kg]
beta_ee = 1.44 ##in [eVnm]
conv = 0.160218 #conversion from [(eVs)^2/(kgeVnm)] to [nm]
alpha = conv*h_bar**2/(2*mass*beta_ee)

r = np.linspace(0,rho_N,num=N)*alpha

#print(eig_val[0],eig_val[1], eig_val[2])
plt.plot(r, X[0]**2, r, X[1]**2, r, X[2]**2, r, X[3]**2)
plt.legend(["omega = 0.01", "omega = 0.5", "omega = 1", "omega = 5"])
plt.xlabel("relative coordinate / nm")
plt.ylabel("probability distribution")
plt.title("probability distribution for the ground state")
#plt.savefig('ground_state',dpi=225)
plt.show()



