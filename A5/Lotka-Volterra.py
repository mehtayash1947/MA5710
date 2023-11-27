import numpy as np
import sys
import scipy
import matplotlib.pyplot as plt

def derivative(X, t, a, b, c, d, fishing):
    x, y = X
    xdash = x * (a - b*y - fishing)
    ydash = y * (-d + c*x - fishing)
    return np.array([xdash, ydash])

def LinearDerivative(dash, X0, t, a, b, c, d, fishing):

    dt = t[1] - t[0]
    Nt = len(t)
    X  = np.zeros([Nt, len(X0)])
    X[0] = X0
    for i in range(Nt - 1):
        X[i+1] = X[i] + dash(X[i], t[i], a,  b, c, d, fishing) * dt

    return X




print("Do you want to solve LV using built-in ODE solver (type 0) or after linearizing it (type 1)?")
mode = int(input())

if mode == 0:
	
	### Solving using built-in ODE solver of scipy
	
	# Parameters
	a = 4
	b = 2
	c = 1.5
	d = 3
	fishing = 0.2
	
	# Initial conditions
	N0 = 10
	P0 = 5
	
	Tmax = 10
	Nsteps = 1000
	t = np.linspace(0, Tmax, Nsteps)
	ICs = [N0, P0]
	res = scipy.integrate.odeint(derivative, ICs, t, args = (a, b, c, d, fishing)) 
	n, p = res.T
	
	plt.figure(0)
	plt.grid(True)
	if fishing != 0:
		plt.title(r'Solution of LV system with ODE solver, $\delta_{fishing} = $' + str(fishing))
	else:
		plt.title("Solution of LV system with ODE solver, no fishing")
		
	plt.plot(t, n, linewidth = 1.5, color = "Green", label = "Deers")
	plt.plot(t, p, linewidth = 1.5, color = "Red", label = "Tigers")
	plt.xlabel(r'Time (days)$\longrightarrow$')
	plt.ylabel(r'Population count$\longrightarrow$')
	plt.legend()
	#plt.savefig("Traj_Nofish.png", dpi = 1500)
	plt.savefig("Traj_fish.png", dpi = 1500)
	
	plt.figure(1)
	plt.grid(True)
	colors = ["Blue", "Red", "Green", "Orange", "Yellow", "Purple"]
#	if fishing != 0:
#		plt.title(r'Phase portraits of LV system with ODE solver, $\delta_{fishing} = $' + str(fishing))
#	else:
#		plt.title("Phase portraits of LV system with ODE solver, no fishing")
		
	#plt.plot(n, p, linewidth = 2, color = 'Blue', label = "N(0) = 10")
#	for j in range(5):
	#N0 = N0 - 1
	plt.title("Phase portrait comparison for initial condition N(0) = 10")
	j = -1
	ICs = [N0, P0]
	resfish = scipy.integrate.odeint(derivative, ICs, t, args = (a, b, c, d, fishing)) 
	nfish, pfish = resfish.T
	res = scipy.integrate.odeint(derivative, ICs, t, args = (a, b, c, d, 0)) 
	n, p = res.T
	plt.plot(n, p , linewidth = 2, color = colors[j + 1], label = "N(0) = " + str(N0) + ", no fishing")
	plt.plot(nfish, pfish, linewidth = 2, color = colors[j + 2], label = "N(0) = " + str(N0) + r', with $\delta_{fishing} = 0.2$')
	
	plt.legend()
	plt.xlabel("Population of Deers")
	plt.ylabel("Population of Tigers")
	#plt.savefig("PP_nofish.png", dpi = 1500)
	plt.savefig("PP_fish_comp.png", dpi = 1500)
	
if mode == 1:
	
	
	
	# Parameters
	a = 4
	b = 2
	c = 1.5
	d = 3
	fishing = 0.2
	
	# Initial conditions
	N0 = 10
	P0 = 5
	
	Tmax = 10
	Nsteps = 1000
	t = np.linspace(0, Tmax, Nsteps)
	ICs = [N0, P0]
	NP = LinearDerivative(derivative, ICs, t, a, b, c, d, fishing)

	plt.figure(0)
	plt.grid(True)
	if fishing != 0:
		plt.title(r'Solution of LV system using Linearization, $\delta_{fishing} = $' + str(fishing))
	else:
		plt.title("Solution of LV system using Linearization, no fishing")
		
	plt.plot(t, NP[:, 0], linewidth = 1.5, color = "Green", label = "Deers")
	plt.plot(t, NP[:, 1], linewidth = 1.5, color = "Red", label = "Tigers")
	plt.xlabel(r'Time (days)$\longrightarrow$')
	plt.ylabel(r'Population count$\longrightarrow$')
	plt.legend()
	#plt.savefig("LinTraj_Nofish.png", dpi = 1500)
	plt.savefig("LinTraj_fish.png", dpi = 1500)


	plt.figure(1)
	colors = ["Blue", "Red", "Green", "Orange", "Yellow", "Purple"]
	#if fishing != 0:
	#	plt.title(r'Phase portraits of LV system using Linearization, $\delta_{fishing} = $' + str(fishing))
	#else:
	#	plt.title("Phase portraits of LV system using Linearization, no fishing")
		

	#for j in range(3):

	plt.title("Phase portrait comparison for initial condition N(0) = 10")
	#j = -1
	ICs = [N0, P0]
	#resfish = scipy.integrate.odeint(derivative, ICs, t, args = (a, b, c, d, fishing)) 
	#nfish, pfish = resfish.T
	resfish = LinearDerivative(derivative, ICs, t, a, b, c, d, fishing) 
	plt.plot(resfish[:, 0], resfish[:, 1] , linewidth = 2, color = "Red", label = "N(0) = " + str(N0) + r', with $\delta_{fishing} = 0.2$')
	res = LinearDerivative(derivative, ICs, t, a, b, c, d, 0)
	plt.plot(res[:, 0], res[:, 1], linewidth = 2, color = "Blue", label = "N(0) = " + str(N0) + ", no fishing")
	#N0 = N0 - 2.5		

	plt.legend()
	plt.xlabel("Population of Deers")
	plt.ylabel("Population of Tigers")
	#plt.savefig("LinPP_nofish.png", dpi = 1500)
	#plt.savefig("LinPP_fish.png", dpi = 1500)
	plt.savefig("LinPP_fish_comp.png", dpi = 1500)
		
	
