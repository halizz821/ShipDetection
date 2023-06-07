# -*- coding: utf-8 -*-
"""
Created on Tue Jun  9 13:31:54 2020

@author: Hamed
"""

# This code detects objects in ocean based on SAR images (back scattering value)
# your image should be prepocessed before using this code (radiometric correction + geometric correction (optional)+ mask all the lands)
# I also provided an sample image 
# The code is implemented using a Two-Parameter Constant False Alarm Rate (CFAR) Detector algorithm, which is also used in SNAP software
# More information about this method can be find in the below paper:
# D. J. Crisp, "The State-of-the-Art in Ship Detection in Synthetic Aperture Radar Imagery." DSTO–RR–0272, 2004-05.
# This code also use paralel processing (using Dask liblary) to speed up the convolution process

import numpy as np
from time import time
import matplotlib.pyplot as plt
# import pywt
from skimage import filters, morphology
from osgeo import gdal

from scipy.ndimage import convolve
from dask import delayed as delayed_dask
from dask import compute



plt.close('all')


start_t = time()



##################### input images
im_name = 'Jao.tif'

dataset = gdal.Open(im_name) # read image using Gdal. You can also use any other library to load your image here
im = np.array(dataset.GetRasterBand(1).ReadAsArray())
#im=filters.gaussian(im,sigma=2) # you can use gaussian filter to reduce the effect of speckle noise


##################### CFAR parameters (refer to the mentioned paper)
kernel_size=31
guard_size=11
target_box_size=3
factor=10  # t parameter in the paper (please read the mentioned paper)




################## define cfar mask

temp=int((kernel_size-guard_size)/2)
mask = np.ones((kernel_size,kernel_size))
mask[temp-1:temp-1+guard_size,temp-1:temp-1+guard_size]=0 # mask consists of train pixes
target_box=np.ones((target_box_size,target_box_size))



############# convolution

ones = np.ones(im.shape)
s = delayed_dask(convolve)(im, mask)
s2 = delayed_dask(convolve)(im**2, mask)
ns = delayed_dask(convolve)(ones, mask)

tbox=delayed_dask(convolve)(im, target_box)

s,s2,ns,tbox=compute(s,s2,ns,tbox)




std=np.sqrt((s2 - s**2 / ns) / ns) #### std of pixels calculated via training pixels

num_train=(kernel_size**2)-(guard_size**2)

im_con=s/num_train  # mean of pixels calculated via training pixels


tbox_mean=tbox/(target_box_size**2)  #mean of pixels calculated in target window

ship_detected=tbox_mean>im_con+factor*std # binary image of the detected objects



print(time()-start_t)


############ identify ship location
# calculate the centroid of each object 

ship_detected=ship_detected.astype('uint8')
ship_detected_la=morphology.label(ship_detected,return_num=False)# classify ships 

from skimage.measure import regionprops
prop=regionprops(ship_detected_la)

ship_loc=[]
for obj_prop in prop:
    ship_loc.append(obj_prop.centroid) # ship centroid
ship_loc=np.asarray(ship_loc)  # Ship locations (x,y) in the image  




################### show image and ships
im = np.abs(10 * np.log10(np.abs(im)))
plt.imshow(im)
plt.plot(ship_loc[:,1],ship_loc[:,0],'r*')
plt.show()

im = np.abs(10 * np.log10(im))
plt.imshow(im)
plt.show()
#####################





