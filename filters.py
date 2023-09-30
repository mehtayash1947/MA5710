import matplotlib.pyplot as plt
import sys
import numpy as np
import math
import scipy
from scipy.optimize import curve_fit
from scipy.ndimage import gaussian_filter
import cv2
from scipy import datasets
import medpy.filter.smoothing as mpf

def psnr(img_noisy, img_cleaned, pmax):

    return 10*np.log10( (pmax**2) / np.mean((img_noisy - img_cleaned)**2) )    


pmax = 255
#modes = ["Gaussian", "Poisson", "SaltPepper", "Speckled"]
#modes = ["Gaussian"]
modes = []

image_file = "bhagirathi-sisters.png"
img = cv2.imread(image_file, 0)
img_f = img.astype(np.float64)


###  This is for adding Gaussian Noise :  ###

mu_temp = 120
sigma_noise_temp = 80
gaussian_noise = np.zeros(img.shape, dtype=np.uint8)
cv2.randn(gaussian_noise, mu_temp, sigma_noise_temp)
gaussian_noise = (0.5*gaussian_noise).astype(np.uint8)
gn_img_temp = cv2.add(img, gaussian_noise)
gn_img_temp_f = gn_img_temp.astype(np.float64)


niter_temp = 10
kappa_temp = 80
gamma_temp = 0.2
sigma_temp = 40

gauss_cleaned_img = gaussian_filter(gn_img_temp, sigma_temp)
gauss_cleaned_img_f = gauss_cleaned_img.astype(np.float64)

PSNR_gauss = psnr(img_f, gauss_cleaned_img_f, pmax)

pmd_img_temp = mpf.anisotropic_diffusion(gn_img_temp, niter=niter_temp, kappa=kappa_temp, gamma = gamma_temp, voxelspacing=None, option=2)
pmd_img_temp_f = pmd_img_temp.astype(np.float64)

PSNR_pmd = psnr(img_f, pmd_img_temp_f, pmax)

print("Gaussian Noise of mu = " + str(mu_temp) + " and sigma = " + str(sigma_noise_temp) + " was added")
print("\n")
print("For Gaussian Cleaning of sigma = " + str(sigma_temp) + ", PSNR = " + str(PSNR_gauss))
print("For Perona-Malik Cleaning of kappa = " + str(kappa_temp) + " and gamma = " + str(gamma_temp) + ", PSNR = " + str(PSNR_pmd))

plt.figure(0)
plt.imshow(pmd_img_temp, cmap = 'gray')
plt.axis("off")
plt.title("Perona Malik Diffusion with " + r'$N_{iter} = $' + str(niter_temp) + r', $\kappa = $' + str(kappa_temp) + r', $\gamma = $' + str(gamma_temp)) 
#plt.savefig("img_perona_malik_" + str(niter) + "_" + str(kappa) + "_" + str(gamma) + ".png", dpi = 1000)
plt.show()

if "Gaussian" in modes:

	PSNR_pmd_10 = []
	PSNR_pmd_25 = []
	PSNR_pmd_75 = []
	PSNR_pmd_100 = []

	PSNR_ld_0p1 = []
	PSNR_ld_1 = []
	PSNR_ld_10 = []
	PSNR_ld_80 = []

	kappa_vals = [10, 25, 75, 100]
	sigma_vals = [0.1, 1, 10, 80]

	mu = 100

	variance_axis = []
	c = 0

	for k in range(5, 100, 2):

		sigma = float(k)
		variance_axis.append(sigma)
		c += 1
		gaussian_noise = np.zeros(img.shape, dtype=np.uint8)
		cv2.randn(gaussian_noise, mu, sigma)
		gaussian_noise = (0.5*gaussian_noise).astype(np.uint8)
		gn_img = cv2.add(img, gaussian_noise)
		gn_img_f = gn_img.astype(np.float64)

		for i in range(0, 4):

			gauss_cleaned_img = gaussian_filter(gn_img, sigma_vals[i])
			gauss_cleaned_img_f = gauss_cleaned_img.astype(np.float64)

			PSNR_gauss = psnr(img_f, gauss_cleaned_img_f, pmax)

			pmd_img = mpf.anisotropic_diffusion(gn_img, niter = niter_temp, kappa = kappa_vals[i], gamma = gamma_temp, voxelspacing=None, option=2)
			pmd_img_f = pmd_img.astype(np.float64)

			PSNR_pmd = psnr(img_f, pmd_img_f, pmax)

			if i == 0:
				PSNR_pmd_10.append(PSNR_pmd)
				PSNR_ld_0p1.append(PSNR_gauss)
			
			if i == 1:
				PSNR_pmd_25.append(PSNR_pmd)
				PSNR_ld_1.append(PSNR_gauss)

			if i == 2:
				PSNR_pmd_75.append(PSNR_pmd)
				PSNR_ld_10.append(PSNR_gauss)

			if i == 3:
				PSNR_pmd_100.append(PSNR_pmd)
				PSNR_ld_80.append(PSNR_gauss)
				
		perc = 100*(float(c)/48)
		print(str(perc) + "% completed")
		
	plt.figure(1)
	plt.plot(variance_axis, PSNR_ld_0p1, linewidth = 0.25, color = 'Black', label = r'LD, $\sigma = 0.1$')
	plt.plot(variance_axis, PSNR_ld_1, linewidth = 0.25, color = 'Blue', label = r'LD, $\sigma = 1$')
	plt.plot(variance_axis, PSNR_ld_10, linewidth = 0.25, color = 'Green', label = r'LD, $\sigma = 10$')
	plt.plot(variance_axis, PSNR_ld_80, linewidth = 0.25, color = 'Red', label = r'LD, $\sigma = 80$')

	plt.plot(variance_axis, PSNR_pmd_10, linewidth = 0.35, linestyle = 'dashed', color = 'Black', label = r'PMD, $\lambda = 10$')
	plt.plot(variance_axis, PSNR_pmd_25, linewidth = 0.35, linestyle = 'dashed', color = 'Blue', label = r'PMD, $\lambda = 25$')
	plt.plot(variance_axis, PSNR_pmd_75, linewidth = 0.35, linestyle = 'dashed', color = 'Green', label = r'PMD, $\lambda = 75$')
	plt.plot(variance_axis, PSNR_pmd_100, linewidth = 0.35, linestyle = 'dashed', color = 'Red', label = r'PMD, $\lambda = 100$')
	plt.legend(loc = 'upper left', prop={'size': 4.5})
	plt.grid(True)
	plt.xlabel(r'Variance $\longrightarrow$')
	plt.ylabel(r'PSNR (in dB) $\longrightarrow$')
	plt.title("PSNR vs. Variance for Gaussian Noise")
	plt.savefig("PSNR_vs_variance_GN.png", dpi = 1000)

if "Poisson" in modes:

	###  For Poission Noise :  ###

	vals = len(np.unique(img))
	vals = 2 ** np.ceil(np.log2(vals))
	poi_noisy_img = np.random.poisson(img * vals) / float(vals)

	PSNR_ld_poi = []
	PSNR_pmd_poi = []
	variance_axis = []

	for y in range(1, 80, 2):

		kappa_poi = float(y)
		sigma_poi = float(y)
		variance_axis.append(y)
		
		gauss_cleaned_img = gaussian_filter(poi_noisy_img, sigma_poi)
		gauss_cleaned_img_f = gauss_cleaned_img.astype(np.float64)

		PSNR_gauss = psnr(img_f, gauss_cleaned_img_f, pmax)

		pmd_img = mpf.anisotropic_diffusion(poi_noisy_img, niter = niter_temp, kappa = kappa_poi, gamma = gamma_temp, voxelspacing=None, option=2)
		pmd_img_f = pmd_img.astype(np.float64)

		PSNR_pmd = psnr(img_f, pmd_img_f, pmax)
		
		PSNR_ld_poi.append(PSNR_gauss)
		PSNR_pmd_poi.append(PSNR_pmd)


		
		
		


	plt.figure(2)
	plt.plot(variance_axis, PSNR_ld_poi, linewidth = 0.25, color = 'Black', label = r'Linear Diffusion')
	plt.plot(variance_axis, PSNR_pmd_poi, linewidth = 0.25, color = 'Blue', label = r'PM Diffusion')

	plt.legend(loc = 'upper left', prop={'size': 6})
	plt.grid(True)
	plt.xlabel(r'$\sigma = \lambda \longrightarrow$')
	plt.ylabel(r'PSNR (in dB) $\longrightarrow$')
	plt.title(r'PSNR vs. $\sigma = \lambda$ for Poission Noise')
	plt.savefig("PSNR_vs_sigma_lambda_Poisson.png", dpi = 1000)


if "Speckled" in modes:

	row,col = img.shape
	gauss = np.random.randn(row,col)
	gauss = gauss.reshape(row,col)        
	speck_noisy_img = img + img*gauss
	
	PSNR_ld_spk = []
	PSNR_pmd_spk = []
	variance_axis = []

	for z in range(1, 150, 2):

		kappa_spk = float(z)
		sigma_spk = float(z)
		variance_axis.append(z)
		
		gauss_cleaned_img = gaussian_filter(speck_noisy_img, sigma_spk)
		gauss_cleaned_img_f = gauss_cleaned_img.astype(np.float64)

		PSNR_gauss = psnr(img_f, gauss_cleaned_img_f, pmax)

		pmd_img = mpf.anisotropic_diffusion(speck_noisy_img, niter = niter_temp, kappa = kappa_spk, gamma = gamma_temp, voxelspacing=None, option=2)
		pmd_img_f = pmd_img.astype(np.float64)

		PSNR_pmd = psnr(img_f, pmd_img_f, pmax)
		
		PSNR_ld_spk.append(PSNR_gauss)
		PSNR_pmd_spk.append(PSNR_pmd)


		
		
		


	plt.figure(3)
	plt.plot(variance_axis, PSNR_ld_spk, linewidth = 0.25, color = 'Black', label = r'Linear Diffusion')
	plt.plot(variance_axis, PSNR_pmd_spk, linewidth = 0.25, color = 'Blue', label = r'PM Diffusion')

	plt.legend(loc = 'upper left', prop={'size': 6})
	plt.grid(True)
	plt.xlabel(r'$\sigma = \lambda \longrightarrow$')
	plt.ylabel(r'PSNR (in dB) $\longrightarrow$')
	plt.title(r'PSNR vs. $\sigma = \lambda$ for Speckled Noise')
	plt.savefig("PSNR_vs_sigma_lambda_Speckle.png", dpi = 1000)
	

if "SaltPepper" in modes:

	row,col = img.shape
	
	
	s2p_ratios = [0.2, 0.5, 0.8]
	amount = 0.004
	
	for p in range(3):
		
		s2p_ratio = s2p_ratios[p]
	
		snp_noisy_img = np.copy(img)
		
		num_salt = np.floor(amount*img.size*s2p_ratio)
		coords = [np.random.randint(0, i - 1, int(num_salt)) 
		for i in img.shape]
		snp_noisy_img[coords] = 1

		num_pepper = np.floor(amount*img.size*(1. - s2p_ratio))
		coords = [np.random.randint(0, j - 1, int(num_pepper)) 
		for j in img.shape]
		snp_noisy_img[coords] = 0
	
		PSNR_ld_snp = []
		PSNR_pmd_snp = []
		variance_axis = []

		for w in range(1, 150, 2):

			kappa_snp = float(w)
			sigma_snp = float(w)
			variance_axis.append(w)
			
			gauss_cleaned_img = gaussian_filter(snp_noisy_img, sigma_snp)
			gauss_cleaned_img_f = gauss_cleaned_img.astype(np.float64)

			PSNR_gauss = psnr(img_f, gauss_cleaned_img_f, pmax)

			pmd_img = mpf.anisotropic_diffusion(snp_noisy_img, niter = niter_temp, kappa = kappa_snp, gamma = gamma_temp, voxelspacing=None, option=2)
			pmd_img_f = pmd_img.astype(np.float64)

			PSNR_pmd = psnr(img_f, pmd_img_f, pmax)
			
			PSNR_ld_snp.append(PSNR_gauss)
			PSNR_pmd_snp.append(PSNR_pmd)


			
		plt.figure(p+4)
		plt.plot(variance_axis, PSNR_ld_snp, linewidth = 0.25, color = 'Black', label = r'Linear Diffusion')
		plt.plot(variance_axis, PSNR_pmd_snp, linewidth = 0.25, color = 'Blue', label = r'PM Diffusion')

		plt.legend(loc = 'upper left', prop={'size': 6})
		plt.grid(True)
		plt.xlabel(r'$\sigma = \lambda \longrightarrow$')
		plt.ylabel(r'PSNR (in dB) $\longrightarrow$')
		plt.title(r'PSNR vs. $\sigma = \lambda$ for Salt n Pepper noise of S/P = ' + str(s2p_ratio))
		plt.savefig("PSNR_vs_sigma_lambda_snp_" + str(s2p_ratio) + ".png", dpi = 1000)

		
		

if len(modes) == 0:

	mu = 100
	sigma = 40
	niter_axis = []
	c = 0
	PSNRs = []
	for k in range(1, 50):

		niter_temp = k
		niter_axis.append(niter_temp)
		c += 1
		gaussian_noise = np.zeros(img.shape, dtype=np.uint8)
		cv2.randn(gaussian_noise, mu, sigma)
		gaussian_noise = (0.5*gaussian_noise).astype(np.uint8)
		gn_img = cv2.add(img, gaussian_noise)
		gn_img_f = gn_img.astype(np.float64)


		pmd_img = mpf.anisotropic_diffusion(gn_img, niter = niter_temp, kappa = 50, gamma = 0.2, voxelspacing=None, option=2)
		pmd_img_f = pmd_img.astype(np.float64)

		PSNR_pmd = psnr(img_f, pmd_img_f, pmax)
		PSNRs.append(PSNR_pmd)
		print("For k = " + str(k) + ", psnr = " + str(PSNR_pmd))

		
	plt.figure(9)
	plt.plot(niter_axis, PSNRs, linewidth = 0.35, color = 'Black')
	plt.grid(True)
	plt.xlabel(r'Number of iterations $\longrightarrow$')
	plt.ylabel(r'PSNR (in dB) $\longrightarrow$')
	plt.title("PSNR vs. Number of iterations for PMD")
	plt.savefig("PSNR_vs_niter_pmd.png", dpi = 1000)






exit() 


if noise_typ =="speckle":
	row,col = image.shape
	gauss = np.random.randn(row,col)
	gauss = gauss.reshape(row,col)        
	noisy = image + image * gauss
	#return noisy



plt.figure(0)
plt.imshow(pmd_img, cmap = 'gray')
plt.axis("off")
plt.title("Perona Malik Diffusion with " + r'$N_{iter} = $' + str(niter) + r', $\kappa = $' + str(kappa) + r', $\gamma = $' + str(gamma)) 
plt.savefig("img_perona_malik_" + str(niter) + "_" + str(kappa) + "_" + str(gamma) + ".png", dpi = 1000)




