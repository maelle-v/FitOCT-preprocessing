# FitOCT-preprocessing
Standardize raw corneal images from a clinical SD-OCT device (RTVue-XR Avanti OCT; Optovue Inc., Fremont, CA, USA) and extract the in-depth attenuation profile from the corneal stroma while compensating for acquisition artifacts.

The output CSV file can be analyzed in terms of Birge ratio and photon mean-free path thanks to Pascal Pernot's FitOCT bayesian procedure, that fits OCT decays using R/stan (2019). The code is available here: https://doi.org/10.5281/zenodo.2579915

The FitOCT pre-processing procedure includes contributions from Romain Bocheux and Bathilde Rivière.

=> The method is currently under publication process. The code, private for now, will be made available for reproducibility purposes.
