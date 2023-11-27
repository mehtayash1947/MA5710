import matplotlib.pyplot as plt
import numpy as np
import scipy
import sys

def perturb(z, rhomax, vmax, e, Nstepsx, Nstepst, K):
	
	for i in range(1, Nstepsx - 1):
		for j in range(0, Nstepst - 1):
			z[i, j + 1] = (vmax/K)*np.log(np.abs(1 + (rhomax/e)*(z[i - 1, j] - z[i, j]))) + z[i,j]
	



xmin = 1
xmax = 20
Nstepsx = int(20)
tmin = 0
tmax = 20
Nstepst = int(21)
x = np.linspace(xmin, xmax, Nstepsx)
t = np.linspace(tmin, tmax, Nstepst)

h = float(x[1] - x[0])
k = float(t[1] - t[0])

z = np.zeros((len(x), len(t)))

for i in range(0, len(t)):
	z[0, i] = 0.1

rhomax = 1
vmax = 1
e = 2.718281
K = 0.28

perturb(z, rhomax, vmax, e, Nstepsx, Nstepst, K)

rho = np.zeros((len(x), len(t)))

for v in range(0, len(t)):
	rho[0, v] = 0.1
	for w in range(1, len(x)):
		rho[w, v] = 1/np.abs(z[w, v] - z[w - 1, v])


print("\t", end = "")	
for i in range(0, Nstepsx - 1):		
	print("x[" + str(i+1) + "]\t", end = "")
	
print("\n")

print(str(t[0]) + "\t", end = "")
#print(str(t[0]) + " & ", end = "")
for kk in range(0,  Nstepsx - 1):

		print(str(round(rho[kk, 0], 4)) + "\t", end = "")


print("\n", end = "")


r = int(Nstepst - 1)
for i in range(1,  r+1) :
	print(str(round(t[i], 2)) + "\t", end = "")
	#print(str(round(t[i*df]/df, 1)) + " & ", end = "")
	
	for j in range(0, Nstepsx - 1):
		q = rho[j, i]	
		print(str(round(q, 4)) + "\t", end = "")	
		#print(str(round(q, 7)) + " & ", end = "")	
	
	print("\n", end = "")
