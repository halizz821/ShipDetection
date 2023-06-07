# ShipDetection
Ship Detection in Ocean using SAR data

This code detects objects in ocean based on SAR images (back scattering value)
- your image should be prepocessed before using this code (radiometric correction + geometric correction (optional)+ mask all the lands)
- I also provided an sample image 
- The code is implemented using a Two-Parameter Constant False Alarm Rate (CFAR) Detector algorithm, which is also used in SNAP software
- More information about this method can be find in the below paper:
 
   D. J. Crisp, "The State-of-the-Art in Ship Detection in Synthetic Aperture Radar Imagery." DSTO–RR–0272, 2004-05.
- This code also use paralel processing (using Dask liblary) to speed up the convolution process
