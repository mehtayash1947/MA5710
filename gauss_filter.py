import matplotlib.pyplot as plt
import sys
import numpy as np
import math
import scipy
from scipy.optimize import curve_fit
from scipy.ndimage import gaussian_filter
import cv2
from scipy import datasets

def psnr(img_noisy, img_cleaned, pmax):

    return 10*np.log10( (pmax**2) / np.mean((img_noisy - img_cleaned)**2) )    



#img = cv2.imread("./kedartal1.png",0)
img = cv2.imread("../A3/bhagirathi-sisters.png", 0)

mu = 120
sigma = 80

gaussian_noise = np.zeros(img.shape, dtype=np.uint8)
cv2.randn(gaussian_noise, mu, sigma)

gaussian_noise = (0.5*gaussian_noise).astype(np.uint8)

gn_img=cv2.add(img, gaussian_noise)


plt.figure(0)
plt.imshow(img,cmap='gray')
plt.axis("off")
plt.title("Original image")
plt.savefig("../A3/img_bw_orig.png", dpi = 1000)

plt.figure(1)
plt.imshow(gn_img,cmap='gray')
plt.axis("off")
plt.title(r'Gaussian noise of $\sigma$ = ' + str(sigma) + " added")
plt.savefig("../A3/img_noisy_" + str(sigma) + "_" + str(mu) + ".png", dpi = 1000)

exit()

pmax = 255	# maximum possible different levels of brightness of a single pixel

img_f = img.astype(np.float64)

sigmas = []
for i in range(10, 401):
	
	sigmas.append(float(i/10))

PSNR_vals = []

for k in range(0, 391):

	cleaned_img = gaussian_filter(gn_img, sigmas[k])

	# Convert images to float for accurate calculations

	cleaned_img_f = cleaned_img.astype(np.float64)
	PSNR = psnr(img_f, cleaned_img_f, pmax)
	PSNR_vals.append(PSNR)


cleaned_img_show_1 = gaussian_filter(gn_img, sigma = 1)
cleaned_img_show_2 = gaussian_filter(gn_img, sigma = 2.5)
cleaned_img_show_3 = gaussian_filter(gn_img, sigma = 0.4)

plt.figure(2)
plt.imshow(cleaned_img_show_1, cmap = 'gray')
plt.axis("off")
plt.title(r'Cleaned image with Gaussian Filter of $\sigma = 1$')
plt.savefig("cleaned_img_sigma_1.png", dpi = 1000)

plt.figure(3)
plt.imshow(cleaned_img_show_2, cmap = 'gray')
plt.axis("off")
plt.title(r'Cleaned image with Gaussian Filter of $\sigma = 2.5$')
plt.savefig("cleaned_img_sigma_2.5.png", dpi = 1000)

plt.figure(4)
plt.imshow(cleaned_img_show_3, cmap = 'gray')
plt.axis("off")
plt.title(r'Cleaned image with Gaussian Filter of $\sigma = 0.4$')
plt.savefig("cleaned_img_sigma_0.4.png", dpi = 1000)


plt.figure(5)
plt.plot(sigmas, PSNR_vals, color = 'Black', linewidth = 1)
plt.xlabel(r'$\sigma \longrightarrow$')
plt.ylabel(r'PSNR (in dB) $\longrightarrow$')
plt.title(r'Plot of PSNR vs. $\sigma$')
plt.grid(True)
plt.savefig("PSNR_vs_sigma.png", dpi = 1000) 






















