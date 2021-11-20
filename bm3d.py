import bm3d
import cv2

# Eventually we can perform sigma_psd optimization on different types of noisy images
# Have to make sure we display what's going on in terminal as these steps can take a 
# ---- long time to process

# Normalizes img at fileName, repeatedly calls compareBM3D() to find optimal sigma_psd
def processImage(fileName):
    return 1.0

# Applies BM3D with initial sigma_psd=0.2,compares sigma_psd=0.1,and sigma_pds=0.3
# ---- by using psnr() and ssim(). specificity, 1 changes sigma by 0.1
# ---- specificty of 2 tells sigma to change by 0.01 (0=tenth,1=hundredth)
def compareBM3D(normalized_noisy_img, original_img, sigma=0.2, specificity=0):
    return 1.0

# Adds noise to an image before/after normalization (not sure yet depends on implementation)
# ---- type -> type of noise
def addNoise(img, type=None):
    return 1.0

# PSNR testing, returns PSNR values from img and post_img comparison 
def psnr(img, post_img):
    return 1.0

# SSIM testing, returns SSIM values from img and post_img comparison
def ssim(img, post_img):
    return 1.0
