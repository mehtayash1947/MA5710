import numpy as np

def run_markov(p, Niter, threshold):
	
	P = np.array([  [1 - p, p, 0, 0], [1 - p, 0, p, 0], [0, 1 - p, 0, p], [0, 0, 1 - p, p]  ])
	
	initial_prob = np.array( [ (1 - p)*(1 - p), p*(1 - p), p*(1 - p), p*p ] )
	
	prob_i = initial_prob
	prob_iprev = -1
	
	for j in range(Niter):
		prob_iprev = prob_i
		prob_i = np.dot(prob_i, P)

	close_flag = np.all(np.abs(prob_iprev - prob_i) < threshold)
		
	return (close_flag, prob_i) 


Bernoulli_parameter = 0.48
Niter = 100
threshold = 1e-6
Res = 2.718

print("Running Markov chain simulation for the following parameters :\n")
print("Bernoulli parameter p = " + str(Bernoulli_parameter))
print("Maximum number of iterations supported = " + str(Niter))
print("Differential threshold for checking convergence = " + str(threshold))

print(".\n.\n.")
for k in range(2, Niter + 1):
	Res = run_markov(Bernoulli_parameter, k, threshold)
	if Res[0] == 1:
		print("Markov chain converged in " + str(k) + " iterations.\nSteady-state probabilities were \n")
		print(Res[1])
		break

if Res[0] == 0:
	print("Markov chain did not converge till " + str(Niter) + " iterations also.")
