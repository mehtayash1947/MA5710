import matplotlib.pyplot as plt
import sys
import numpy as np
import math
import scipy
from scipy.optimize import curve_fit


alpha = 0.1
cubicity = pow(np.pi/4 , (1/3))


R_exp = int(sys.argv[1])
h_exp = int(sys.argv[2])
mode = int(sys.argv[3])

def pack_lb(R, h, r):
	
	sum = 0
	j = 1
	if 2*r > R:
		return math.floor(h/(2*r))
		
	while True:
		
		denom = R - j*r
		if denom < r:
			break
		sum += math.floor((1/math.atan(r/denom)))
		j += 2
	
	return math.floor((np.pi)*math.floor(h/(2*r))*sum)

def pack_fcc_ub(R, h, r):
	
	return (((np.pi)*R*R*h)/( 4*np.sqrt(2)*r*r*r))


def model(r, c0, c1, c2, c3):
	
	return c0 + (c1/r) + (c2/(r*r)) + (c3/(r*r*r))


if mode == 2:

	print("***You are executing the script assuming the model parameter file is ready and has the values corresponding to the ones you want.***")
	R_exp = int(input("Enter value of R (Radius of Cylinder)\n"))
	h_exp = int(input("Enter value of h (Height of Cylinder)\n"))
	r_in = float(input("Enter value of r (Radius of Cherry)\n"))
	params_read = open("./model_params.txt", "r")
	all_lines = params_read.readlines()
	found_flag = 0
	ave_no = 1
	R_prev = R_exp - (R_exp % 5)
	h_prev = h_exp - (h_exp % 5)
	R_next = R_prev + 5
	h_next = h_prev + 5
	if R_exp != R_prev:
		ave_no = ave_no*2
	if h_exp != h_prev:
		ave_no = ave_no*2
	   
	
	for line in all_lines:
		
		# for any random R, h say 996, 506, do distance-weighted average of (995,505), (1000,505), (995,510) and (1000,510) and use x%5 for simplicity
		words = line.split('\t')
		
		if ave_no == 1:
			
			if int(words[0]) == R_exp and int(words[1]) == h_exp:
				
				found_flag = 1
				c0 = float(words[2])
				c1 = float(words[3])
				c2 = float(words[4])
				c3 = float(words[5])
				rp = int(0.50 + model(r_in, c0, c1, c2, c3))
				print("Number of cherries that can be packed is " + str(rp))
				break
			
		if ave_no == 2:
		
			rp = 0.0
			if int(words[0]) == R_exp and int(words[1]) == h_prev:
				
				c0 = float(words[2])
				c1 = float(words[3])
				c2 = float(words[4])
				c3 = float(words[5])
			
				rp += int(0.50 + model(r_in, c0, c1, c2, c3)*(h_next - h_exp)/5)
				
			if int(words[0]) == R_exp and int(words[1]) == h_next:
				
				c0 = float(words[2])
				c1 = float(words[3])
				c2 = float(words[4])
				c3 = float(words[5])
			
				rp += int(0.50 + model(r_in, c0, c1, c2, c3)*(h_exp - h_prev)/5 )
			  
			if int(words[1]) == h_exp and int(words[0]) == R_prev:
				
				c0 = float(words[2])
				c1 = float(words[3])
				c2 = float(words[4])
				c3 = float(words[5])
			
				rp += int(0.50 + model(r_in, c0, c1, c2, c3)*(R_next - R_exp)/5)

			if int(words[1]) == h_exp and int(words[1]) == R_next:
				
				c0 = float(words[2])
				c1 = float(words[3])
				c2 = float(words[4])
				c3 = float(words[5])
			
				rp += int(0.50 + model(r_in, c0, c1, c2, c3)*(R_exp - R_prev)/5)
				
			
			print("Number of cherries that can be packed is " + str(rp))
			break

		if ave_no == 4:
				
			rp = 0.0
			if int(words[0]) == R_prev and int(words[1]) == h_prev:
				
				c0 = float(words[2])
				c1 = float(words[3])
				c2 = float(words[4])
				c3 = float(words[5])
			
				rp += int(0.50 + model(r_in, c0, c1, c2, c3)*(h_next - h_exp)/5)
				
			if int(words[0]) == R_prev and int(words[1]) == h_next:
				
				c0 = float(words[2])
				c1 = float(words[3])
				c2 = float(words[4])
				c3 = float(words[5])
			
				rp += int(0.50 + model(r_in, c0, c1, c2, c3)*(h_exp - h_prev)/5)
			  
			if int(words[1]) == h_prev and int(words[0]) == R_prev:
				
				c0 = float(words[2])
				c1 = float(words[3])
				c2 = float(words[4])
				c3 = float(words[5])
			
				rp += int(0.50 + model(r_in, c0, c1, c2, c3)*(R_next - R_exp)/5)

			if int(words[1]) == h_prev and int(words[1]) == R_next:
				
				c0 = float(words[2])
				c1 = float(words[3])
				c2 = float(words[4])
				c3 = float(words[5])
			
				rp += int(0.50 + model(r_in, c0, c1, c2, c3)*(R_exp - R_prev)/5)
				
			
			print("Number of cherries that can be packed is " + str(rp))
			break
						
			


	
	
	if found_flag == 0:
		print("No entry found in model_params file correspinding to given R and h")

	exit()



N_ub = np.array([])
N_lb = np.array([])
N_sp_corrected = np.array([])
N_tbf = np.array([])
error = np.array([])
r_exp = np.array([])

start = 1 + (1 - mode)*19

for r in range(start, 1 + int(min(R_exp, h_exp/2))):

	r_exp = np.append(r_exp, [int(r)])
	
	Nu = pack_fcc_ub(R_exp, h_exp, r)
	Nl = pack_lb(R_exp, h_exp, r)
	
	
	if mode == 0:
		N_ub = np.append(N_ub, [Nu])
		N_lb = np.append(N_lb, [Nl])
		error = np.append(error, [(Nu - Nl)])
		N_sp_corrected = np.append(N_sp_corrected, [Nu*cubicity])
	
	if (Nl/(1 - alpha)) > (Nu/(1 + alpha)): 
		N_tbf = np.append(N_tbf, [math.floor((Nu/(1 + alpha)))])
		print("Entered if")
	else:
		N_tbf = np.append(N_tbf, [math.floor(Nl/(1 - alpha))])

		
if mode == 1:

	fp, _ = curve_fit(model, r_exp, N_tbf)
	
	model_file = open("./model_params.txt", "a")
	model_file.write(str(R_exp) + "\t" + str(h_exp) + "\t" + str(fp[0]) + "\t" + str(fp[1]) + "\t" + str(fp[2]) + "\t" + str(fp[3]) + "\n")
	


if mode == 0:



	plt.figure(0)
	plt.plot(r_exp, N_lb, color = 'Green', linewidth = 0.3, label = 'Lower bound')
	plt.plot(r_exp, N_ub, color = 'Red', linewidth = 0.3, label = 'Upper bound (using FCC)')
	plt.plot(r_exp, N_sp_corrected, color = 'Blue', linewidth = 0.3, label = 'FCC Upper bound (Cubicity corrected)')
	plt.xlabel(r'Radius of cherry $\longrightarrow$')
	plt.ylabel(r'Number of cherries that can be packed')
	plt.title(r'Comparison of packing for R = ' + str(R_exp) + ", h = " + str(h_exp) )
	plt.legend()
	plt.grid(True)
	plt.savefig("estimate_comparison.png", dpi = 2000)
	
	fp, _ = curve_fit(model, r_exp, N_tbf)
	plt.figure(1)
	plt.plot(r_exp, N_tbf, color = 'Blue', linewidth = 0.5, label = 'N (to be fitted)')
	plt.plot(r_exp, model(r_exp, fp[0], fp[1], fp[2], fp[3]), color = 'Red', linewidth = 0.50, label = 'Curve-fitted model')
	plt.xlabel(r'Radius of cherry $\longrightarrow$')
	plt.ylabel(r'Number of cherries that can be packed')
	plt.title(r'Fitting model to curve')
	plt.legend()
	plt.grid(True)
	plt.savefig("curve_fitting.png", dpi = 2000)
	


			
	
	


