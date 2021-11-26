# Fall 2021 CIS 465 Term Project
## By: Chris Sweeney and Dakotah Pettry
Analysis of BM3D Image processing technique on various noise types, with sigma optimization

## Instructions
Make sure you install all the required python3 libraries
- bm3d
- cv2
- numpy
- skimage
Sample Instructions
```py
processImage('Lenna.png',['gaussian'],starting_sigma=0.1, starting_noise_variance=0.12)
```
Parameters
1. file name of image to process
2. list of noise types to process, support gaussian and speckle at the moment
3. optional starting_sigma
4. optional starting_noise_variance to specify intensity of noise

