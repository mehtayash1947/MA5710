import matplotlib.pyplot as plt
import numpy as np
import scipy
import sys

### v = (1 - rho)
### rho_t + (rho*v)_x = 0
### rho_t + rho_x - 2*rho*rho_x = 0
### u = 1 - 2*rho
### u_t + (u^2/2)_x = -2*rho_t + u*u_x = -2*rho_t + (1 - 2*rho)(-2*rho_x) = -2*rho_t - 2*rho_x + 4*rho*rho_x = 0 => rho_t + rho_x - 2*rho*rho_x = 0

def f(rho, rhomax, vmax):
	return vmax*rho*(1 - rho/rhomax)

def v(rho, rhomax, vmax):
	return vmax*(1 - rho/rhomax)

def f_dash(rho, rhomax, vmax):
	return vmax*(1 - 2*rho/rhomax)

def f1(rho, rhocrit, rhojam, K):
	
	if rho < rhocrit:
		return K*rho*(1 - rhocrit/rhojam)
	else:
		return rho*(K - (K*rho/rhojam))

def v1(rho, rhocrit, rhojam, K):
	if rho < rhocrit:
		return K*(1 - rhocrit/rhojam)
	else:
		return K - K*rho/rhojam

def f1_dash(rho, rhocrit, rhojam, K):
	
	if rho < rhocrit:
		return 0
	else:
		return K*(1 - 2*rho/rhojam)
		

def solve(rho, ix, it, Nstepst, Nstepsx, rhomax, vmax):


	for ixx in range(1, Nstepsx - 1):
		
		if f_dash(rho[ixx, it], rhomax, vmax) == 0:
			return (rho[ixx, it], rho[ixx - 1, it])

	for t in range(0, Nstepst - 1):
	
		for ix2 in range(1, Nstepsx - 1):
			
			if f_dash(rho[ix2, t], rhomax, vmax) == 0:
				return (rho[ix2, t], rho[ix2 - 1, t])

	return (rhomax/2, rhomax/2)
	
	print("Solver couldn't solve")

	return (rho[ix, it], rho[ix - 1, it])
	#return (rho[ix + 1, it], rho[ix, it])

def solve1(rho, ix, it, Nstepst, Nstepsx, rhocrit, rhojam, K):


	for ixx in range(1, Nstepsx - 1):
		
		if f1_dash(rho[ixx, it], rhocrit, rhojam, K) == 0:
			return (rho[ixx, it], rho[ixx - 1, it])

	for t in range(0, Nstepst - 1):
	
		for ix2 in range(1, Nstepsx - 1):
			
			if f1_dash(rho[ix2, t], rhocrit, rhojam, K) == 0:
				return (rho[ix2, t], rho[ix2 - 1, t])

	
	return (rhojam/2, rhojam/2)
				
	print("Solver couldn't solve")

	return (rho[ix, it], rho[ix - 1, it])
	#return (rho[ix + 1, it], rho[ix, it])


def Gudanov(rho, Nstepst, Nstepsx, rhomax, vmax, k, h):

	for it in range(0, Nstepst - 1):
		
		for ix in range(0, Nstepsx - 1):
			
			ui = -1	
			uiminus = -1			## Initialised to -1 to detect error (in case they don't get updated)
			
	## If cond 1

			if f_dash(rho[ix, it], rhomax, vmax) >= 0 and f_dash(rho[ix + 1, it], rhomax, vmax) >= 0:
				ui = rho[ix, it]
				uiminus = rho[ix - 1, it]

	## If cond 2
				
			if f_dash(rho[ix, it], rhomax, vmax) < 0 and f_dash(rho[ix + 1, it], rhomax, vmax) < 0:
				ui = rho[ix + 1, it]
				uiminus = rho[ix, it]

	## If cond 3
		
			if f_dash(rho[ix, it], rhomax, vmax) >= 0 and f_dash(rho[ix + 1, it], rhomax, vmax) < 0:
				s = ( f(rho[ix + 1, it], rhomax, vmax) - f(rho[ix, it], rhomax, vmax) )/(rho[ix + 1, it] - rho[ix, it])
				if s >= 0:
					ui = rho[ix, it]
					uiminus = rho[ix - 1, it]
				else:
					ui = rho[ix + 1, it]
					uiminus = rho[ix, it]

	## If cond 4
			
			if f_dash(rho[ix, it], rhomax, vmax) < 0 and f_dash(rho[ix + 1, it], rhomax, vmax) >= 0:
				(ui, uiminus) = solve(rho, ix, it, Nstepst, Nstepsx, rhomax, vmax)

				if ui == -1:
					print("Solver couldn't find a solution for equation j_dash(x, t) = 0 at t = " + str(it) + ". Exiting program..." )
					exit(1)
			
			if ui == -1 or uiminus == -1:
				print("ui did not get updated for x = " + str(ix) + ", t = " + str(it) + ". Exiting program...\n")
				exit(1)
			
			### Updation:
			
			rho[ix, it + 1] = rho[ix, it] - (k/h)*( f(ui, rhomax, vmax) - f(uiminus, rhomax, vmax) )
			#print("No issue in updating ix = " + str(ix) + " and it = " + str(it))


def Gudanov1(rho, Nstepst, Nstepsx, rhocrit, rhojam, K, k, h):


	for it in range(0, Nstepst - 1):
		
		for ix in range(0, Nstepsx - 1):
			#if ix == 6:
			#	rho[ix, it] = 1.0
			#	continue
			#if ix > 6:
			#	rho[ix, it] = 0
			#	continue
							
			ui = -1	
			uiminus = -1			## Initialised to -1 to detect error (in case they don't get updated)
			
	## If cond 1

			if f1_dash(rho[ix, it], rhocrit, rhojam, K) >= 0 and f1_dash(rho[ix + 1, it], rhocrit, rhojam, K) >= 0:
				ui = rho[ix, it]
				uiminus = rho[ix - 1, it]

	## If cond 2
				
			if f1_dash(rho[ix, it], rhocrit, rhojam, K) < 0 and f1_dash(rho[ix + 1, it], rhocrit, rhojam, K) < 0:
				ui = rho[ix + 1, it]
				uiminus = rho[ix, it]

	## If cond 3
		
			if f1_dash(rho[ix, it], rhocrit, rhojam, K) >= 0 and f1_dash(rho[ix + 1, it], rhocrit, rhojam, K) < 0:
				s = ( f1(rho[ix + 1, it], rhocrit, rhojam, K) - f1(rho[ix, it], rhocrit, rhojam, K) )/(rho[ix + 1, it] - rho[ix, it])
				if s >= 0:
					ui = rho[ix, it]
					uiminus = rho[ix - 1, it]
				else:
					ui = rho[ix + 1, it]
					uiminus = rho[ix, it]

	## If cond 4
			
			if f1_dash(rho[ix, it], rhocrit, rhojam, K) < 0 and f1_dash(rho[ix + 1, it], rhocrit, rhojam, K) >= 0:
				(ui, uiminus) = solve1(rho, ix, it, Nstepst, Nstepsx, rhocrit, rhojam, K)

				if ui == -1:
					print("Solver couldn't find a solution for equation j_dash(x, t) = 0 at t = " + str(it) + ". Exiting program..." )
					exit(1)
			
			if ui == -1 or uiminus == -1:
				print("ui did not get updated for x = " + str(ix) + ", t = " + str(it) + ". Exiting program...\n")
				exit(1)
			
			### Updation:
			
			rho[ix, it + 1] = rho[ix, it] - (k/h)*( f1(ui, rhocrit, rhojam, K) - f1(uiminus, rhocrit, rhojam, K) )
			#if rho[ix, it+1] < 0:
				#rho[ix, it+1] = rho[ix - 1, it+1]



rhomax = 1
vmax = 1
rhocrit = 0.05
rhojam = 0.99
K = 1.0532

X = np.linspace(0, 1, 200)
Y1 = np.zeros(200)
Y2 = np.zeros(200)
for yy in range(0, 200):
	Y1[yy] = v(X[yy], rhomax, vmax)
	Y2[yy] = v1(X[yy], rhocrit, rhojam, K)
plt.figure(0)
plt.plot(X, Y1, linewidth = 1, color = 'Blue', label = 'Relation 1')
plt.plot(X, Y2, linewidth = 1, color = 'Red', label = 'Relation 2')
plt.xlabel(r'$\rho\longrightarrow$')
plt.ylabel(r'$v$ (velocity)')
plt.legend()
plt.title("DIfferent speed density relations used")
plt.savefig("SDR.png", dpi = 1000)
#exit(1)




xmin = 1
xmax = 100
Nstepsx = int(100)
xdmax = 15
tmin = 0
tmax = 5
df = 10
Nstepst = 51

red = 6
tred = 0.5
tgreen = 1.0
tredagain = 1.5

x = np.linspace(xmin, xmax, Nstepsx)
t = np.linspace(tmin, tmax, Nstepst)

h = float(x[1] - x[0])
k = float(t[1] - t[0])



rho = np.zeros((len(x), len(t)))

for ixi in range(0, len(x)):

		if ixi <= red:
			rho[ixi,0] = 0.4*rhomax
		#if ixi == red:
			#rho[ixi, 0] = rhomax
		if ixi > red:
			rho[ixi, 0] = 0


#Gudanov(rho, Nstepst, Nstepsx, rhomax, vmax, k, h)

Gudanov1(rho, Nstepst, Nstepsx, rhocrit, rhojam, K, k, h)

	
#for mz in range(0, Nstepst):
#	rho[red, mz] = 0
					
			
print("\t", end = "")	
for i in range(0, min(xdmax, Nstepsx - 1)):
	if i >= red:
		print("x[" + str(i+1) + "]\t", end = "")
		continue
				
	print("x[" + str(i+1) + "]\t", end = "")
	
print("\n")

print(str(t[0]) + "\t", end = "")
#print(str(t[0]) + " & ", end = "")
for kk in range(0, min(xdmax, Nstepsx - 1)):

	if kk >= red:
		print(str(round(rho[kk, 0], 4)) + "\t", end = "")
		#print(str(rho[kk, 0]) + " & ", end = "")
	
		continue
	
	print(str(rho[kk, 0]) + "\t", end = "")
	#print(str(rho[kk, 0]) + " & ", end = "")

print("\n", end = "")


r = int((Nstepst - 1)/df)
for i in range(1, r+1):
	print(str(round(t[i*df]/df, 2)) + "\t", end = "")
	#print(str(round(t[i*df]/df, 1)) + " & ", end = "")
	
	for j in range(0, min(xdmax, Nstepsx - 1)):
		q = rho[j, i*df]	
		print(str(round(q, 4)) + "\t", end = "")	
		#print(str(round(q, 7)) + " & ", end = "")	
	
	print("\n", end = "")

green_flag = 0
if green_flag == 1:

	df = 5
	Nstepstnew = 5*df + 1
	#Nstepstnew = 101
	rho2 = np.zeros((len(x), Nstepstnew))

	for w in range(0, Nstepsx):
		rho2[w, 0] = rho[w, Nstepst - 1]
	tnew = np.linspace(0.5, 1, Nstepstnew)
	#print("k and h were " + str(k) +  " and " + str(h))
	Gudanov(rho2, Nstepstnew, Nstepsx, rhomax, vmax, k, h)
	print("\n\nSignal turns green at t = 0.5 \n")
	r = int((Nstepstnew - 1)/df)
	#print("r = " + str(r))
	for i in range(0, r+1):
		#print(str(round(tnew[i*df] , 2))+ " \t", end = "")
		print(str(round(t[i*df]/df, 1)) + " & ", end = "")
		for j in range(0, min(xdmax, Nstepsx - 1)):
			q = rho2[j, i*df]	
			#print(str(round(q, 7)) + " \t", end = "")	
			print(str(round(q, 7)) + " & ", end = "")			
		print("\n", end = "")

red_flag = 0
if red_flag == 1:
	
	df = 5
	Nstepstnew = 5*df + 1
	rho3 = np.zeros((len(x), Nstepstnew))
	for v in range(0, Nstepsx):
		rho3[v, 0] = rho2[v, Nstepstnew - 1]
	
	for mz in range(0, Nstepstnew):
		rho3[red, mz] = 1.0

	tnew = np.linspace(1, 1.5, Nstepstnew)
	
	Gudanov(rho3, Nstepstnew, Nstepsx, rhomax, vmax, k, h)

	print("\nSignal turns red again at t = 1 \n")
	r = int((Nstepstnew - 1)/df)
	
	for s in range(0, r+1):
		#print(str(round(tnew[s*df], 2)) + "\t", end = "")
		print(str(round(t[i*df]/df, 1)) + " & ", end = "")
		for j in range(0, min(xdmax, Nstepsx - 1)):
			if j == red:
				continue
			q = rho3[j, s*df]
			#print(str(round(q, 7)) + "\t", end = "")
			print(str(round(q, 7)) + " & ", end = "")			
		print("\n", end = "")
