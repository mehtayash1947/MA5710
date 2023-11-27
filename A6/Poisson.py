import numpy as np
import matplotlib.pyplot as plt

transition_matrix = np.array( [  [0.98, 0.02],   [0.99, 0.01]  ] )

t = 0 
T_total = 1000  

state_array = [0]
t_array = [0]


state_i = state_array[0]
for t in range(T_total):

    lambdas = transition_matrix[state_i]
    lambda_total = sum(lambdas)
    
    if lambda_total == 0:
        break

    dt = np.random.exponential(scale=1/lambda_total)
    t += dt    
   
    state_iplus1 = np.random.choice([0, 1], p = lambdas/lambda_total)
    
    state_i = state_iplus1
    state_array.append(state_i)
    t_array.append(t)


plt.step(t_array, state_array, linewidth = 0.3, color = 'Red')
plt.xlabel(r'Time $\longrightarrow$')
plt.ylabel(r'State $\longrightarrow$')
plt.title(r'Simulation of Poisson Process, p (0 to 1) = 0.02')
plt.savefig("Poisson_simu_0.98.png", dpi = 1500)
