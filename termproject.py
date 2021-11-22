import bm3d
import cv2
import numpy as np
from skimage.util import noise, random_noise

# Eventually we can perform sigma_psd optimization on different types of noisy images
# Have to make sure we display what's going on in terminal as these steps can take a 
# ---- long time to process

# Normalizes img at fileName, repeatedly calls compareBM3D() to find optimal sigma_psd
def processImage(file_name,noise_types):
    # read in original image
    img = cv2.imread(file_name)

    # apply random_noise (automatically normalizes image)
    # ---- mode -> gaussian, s&p, possion, or speckle
    noisy_images = addNoise(img,noise_types=noise_types,variance=0.05)

    for index,noisy_image in enumerate(noisy_images):
        compareBM3D(noisy_image,img,sigma=0.04,noise_type=noise_types[index])

    return 1.0

# Applies BM3D with initial sigma_psd=0.2,compares sigma_psd=0.1,and sigma_pds=0.3
# ---- by using psnr() and ssim(). specificity, 1 changes sigma by 0.1
# ---- specificty of 1 tells sigma to change by 0.01 (0=tenth,1=hundredth)
# ---- stage_arg -> HARD_THRESHOLDING or ALL_STAGES (slower but better)
def compareBM3D(normalized_noisy_img, original_img, sigma=0.2, specificity=0,noise_type=''):
    lesser_sigma = sigma-(0.1-(specificity*0.09))
    greater_sigma = sigma+(0.1-(specificity*0.09))

    print('Processing '+noise_type)
    lesser_img = bm3d.bm3d(normalized_noisy_img, sigma_psd=lesser_sigma,stage_arg=bm3d.BM3DStages.ALL_STAGES)
    img = bm3d.bm3d(normalized_noisy_img, sigma_psd=sigma,stage_arg=bm3d.BM3DStages.ALL_STAGES)
    greater_img = bm3d.bm3d(normalized_noisy_img, sigma_psd=greater_sigma,stage_arg=bm3d.BM3DStages.ALL_STAGES)
    print('Finished, close window to continue')
    
    # Compare PSNR and SSIM test results for each variation 

    if True:
        # original_img needs to be normalized to display with cv2.imshow()
        norm_image = cv2.normalize(original_img, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
        cv2.imshow(
            'Before | '+noise_type+' | After',
            np.concatenate((norm_image,normalized_noisy_img,img),axis=1)
            )
        cv2.waitKey()


# Adds noise to an image before/after normalization (not sure yet depends on implementation)
# ---- type -> type of noise
def addNoise(img, noise_types=['gaussian'],variance=0.2):
    noisy_images = []
    for noise_type in noise_types:
        noisy_images.append(random_noise(img,mode=noise_type,var=variance**2))
    return noisy_images

# PSNR testing, returns PSNR values from img and post_img comparison 
def psnr(img, post_img):
    return 1.0

# SSIM testing, returns SSIM values from img and post_img comparison
def ssim(img, post_img):
    return 1.0

processImage('Lenna.png',['gaussian','speckle'])
