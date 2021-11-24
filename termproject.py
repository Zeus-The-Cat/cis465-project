import bm3d
import cv2
import numpy as np
from skimage.util import noise, random_noise
from skimage.metrics import structural_similarity

# Eventually we can perform sigma_psd optimization on different types of noisy images
# Have to make sure we display what's going on in terminal as these steps can take a
# ---- long time to process

# Normalizes img at fileName, repeatedly calls compareBM3D() to find optimal sigma_psd
def processImage(file_name,noise_types,starting_sigma=0.2,starting_noise_variance=0.1):
    # read in original image
    img = cv2.imread(file_name)

    # apply random_noise (automatically normalizes image)
    # ---- mode -> gaussian, s&p, possion, or speckle
    noisy_images = addNoise(img,noise_types=noise_types,variance=starting_noise_variance)

    for index,noisy_image in enumerate(noisy_images):
        print('Processing '+noise_types[index]+' noise at '+str(starting_noise_variance) +' variance')
        fimg = compareBM3D(noisy_image,img,sigma=starting_sigma,noise_type=noise_types[index])
        cv2.imwrite(noise_types[index]+str(starting_noise_variance)+'_sig'+fimg[1]+'_'+file_name,fimg[0])
    return 1.0

# Applies BM3D with initial sigma_psd=0.2,compares sigma_psd=0.1,and sigma_pds=0.3
# ---- by using psnr() and ssim(). specificity, 1 changes sigma by 0.1
# ---- specificty of 1 tells sigma to change by 0.01 (0=tenth,1=hundredth)
# ---- stage_arg -> HARD_THRESHOLDING or ALL_STAGES (slower but better)
def compareBM3D(normalized_noisy_img, original_img, sigma=0.2, specificity=0,noise_type=''):
    specificity_value = round(0.1-specificity*0.09,2)
    lesser_sigma = sigma-specificity_value
    greater_sigma = sigma+specificity_value
    if greater_sigma > 1.0:
        greater_sigma = 1.0
    if lesser_sigma <= 0:
        lesser_sigma = 0.01

    print('Processing BM3D (1/3)',end='\r')
    lesser_img = bm3d.bm3d(normalized_noisy_img, sigma_psd=lesser_sigma,stage_arg=bm3d.BM3DStages.ALL_STAGES)
    print('Processing BM3D (2/3)',end='\r')
    img = bm3d.bm3d(normalized_noisy_img, sigma_psd=sigma,stage_arg=bm3d.BM3DStages.ALL_STAGES)
    print('Processing BM3D (3/3)',end='\r')
    greater_img = bm3d.bm3d(normalized_noisy_img, sigma_psd=greater_sigma,stage_arg=bm3d.BM3DStages.ALL_STAGES)
    print('Processing BM3D (3/3) Finished')

    # Compare PSNR and SSIM test results for each variation with terminal feedback
    norm_image = cv2.normalize(original_img, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)

    ssim_lesser = calc_ssim(norm_image,lesser_img)
    ssim_starting = calc_ssim(norm_image,img)
    ssim_greater = calc_ssim(norm_image,greater_img)
    print('SSIM value at '+str(round(lesser_sigma,2))+' sigma:'+str(ssim_lesser))
    print('SSIM value at '+str(round(sigma,4))+' sigma:'+str(ssim_starting))
    print('SSIM value at '+str(round(greater_sigma,4))+' sigma:'+str(ssim_greater))

    display_results = False
    greatest = compareSSIM(ssim_lesser,ssim_starting,ssim_greater)
    if greatest == 0:
        # search finished or need to increase specificity
        if specificity == 1:
            print('Best Sigma '+str(round(sigma,4))+' for '+str(noise_type))
            display_results = True
        if specificity == 0:
            print('Increasing Precision')
            return compareBM3D(normalized_noisy_img,original_img,sigma=sigma,specificity=1,noise_type=noise_type)
    elif greatest == 1:
        print('Increasing sigma by '+str(specificity_value))
        return compareBM3D(normalized_noisy_img,original_img,sigma=greater_sigma,specificity=specificity,noise_type=noise_type)
    else:
        print('Decreasing sigma by '+str(specificity_value))
        return compareBM3D(normalized_noisy_img,original_img,sigma=lesser_sigma,specificity=specificity,noise_type=noise_type)


    if display_results:
        print('Finished, close window to continue')
        cv2.imshow(
            'Before | '+noise_type+' | BM3D '+str(round(sigma,2)),
            np.concatenate((norm_image,normalized_noisy_img,img),axis=1)
            )
        cv2.waitKey()
        img_out = np.concatenate((norm_image,normalized_noisy_img,img),axis=1)
        img_out = img_out*255
        img_out = img_out.astype('uint8')
        return (img_out , str(round(sigma,2)))

# returns SSIM with most similarity to original
def compareSSIM(less,origin,greater):
    if less > origin and less >= greater:
        return -1
    if origin >= less and origin >= greater:
        return 0
    return 1

# Adds noise to an image before/after normalization (not sure yet depends on implementation)
# ---- type -> type of noise
def addNoise(img, noise_types=['gaussian'],variance=0.2):
    noisy_images = []
    for noise_type in noise_types:
        noisy_images.append(random_noise(img,mode=noise_type,var=variance**2))
    return noisy_images

#This site made these pretty trivial https://cvnote.ddlee.cc/2019/09/12/psnr-ssim-python
# am smol bran, let me know if i can try to fix this, if no work good.
# PSNR testing, returns PSNR values from img and post_img comparison
def psnr(img, post_img):
    psnr = cv2.PSNR(img, post_img)
    return psnr

#this calls ssim, to stay consistent with the implementation I'm basing all of this off of
#should this maybe call compareBM3D rather than ssim?
def calc_ssim(img, post_img):
    if (img.shape == post_img.shape):
        if(img.ndim == 2):
            return ssim(img, post_img)
        elif(img.ndim == 3):
            if(img.shape[2] == 3):
                _ssim = []
                for i in range(3):
                    _ssim.append(ssim(img,post_img))
                return np.array(_ssim).mean()
            elif(img.shape[2]==1):
                return ssim(np.squeeze(img), np.squeeze(post_img))
        else:
            return("error with image dimesions in calc_ssim")

# SSIM testing, returns SSIM values from img and post_img comparison
# does the sigma value need to be changed here?
def ssim(img, post_img):
    C1 = (0.01 * 255) ** 2
    C2 = (0.03 * 255) ** 2

    gkernal = cv2.getGaussianKernel(11, 1.5)
    window = np.outer(gkernal, gkernal.transpose())

    mu1 = cv2.filter2D(img, -1, window)[5:-5, 5:-5]
    mu2 = cv2.filter2D(post_img, -1, window)[5:-5, 5:-5]
    sigma1_sq = cv2.filter2D(img**2, -1, window)[5:-5, 5:-5] - (mu1**2)
    sigma2_sq = cv2.filter2D(post_img**2, -1, window)[5:-5, 5:-5] - (mu2**2)
    sigma12 = cv2.filter2D(img * post_img, -1, window)[5:-5, 5:-5] - (mu1*mu2)

    ssim_map = ((2 * mu1* mu2 + C1) * (2 * sigma12 + C2)) / ((mu1**2 + mu2**2 + C1) * (sigma1_sq + sigma2_sq + C2))
    return ssim_map.mean()

# Super Noisy image low detail retention
# processImage('Lenna.png',['gaussian'],starting_noise_variance=0.75)
# Medium Noise levels Blobby but maintained areas of interest
# processImage('Baboon.png',['gaussian'],starting_noise_variance=0.5)
# Low Noise Levels nearly identical, with few minor details missing
# processImage('peppers.jpg',['gaussian'],starting_sigma=0.1, starting_noise_variance=0.1)


# Demo
processImage('Lenna.png',['gaussian'],starting_sigma=0.1, starting_noise_variance=0.05)
