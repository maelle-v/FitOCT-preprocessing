# -*- coding: utf-8 -*-
"""
Created on Mon Feb 21 12:12:07 2022

@author: Maëlle
Return the preprocessed SD-OCT corneal images (flattened + corrected for 
artifacts) acquired with RTVue-XR Avanti SD-OCT device by Optovue Inc.
"""

#%% USER INPUT

#pathname = 'C:\\Users\\Maëlle\\Documents\\Rédaction\\_Thèse\\ThesisTemplate-Maelle\\Figures\\supplémentaires\\_preproc\\LADFED_OD' # chemin du dossier où est stockée l'image
pathname = "C:\\Users\\Maëlle\\Documents\\Code - SD-OCT études\\__démo codes\\Line\\"
filename = '_VILBERT_MAELLE__89337_Cornea Line_OD_2020-10-14_11-17-55_F_1994-02-07_1.jpg' # nom original de l'image

champ_analyse_mm = 6        # [mm] width of analyzed area (comment the line if analyzing the full image width)
marge_postBowman = 50       # [µm] post-Bowman layer z-margin [default 50 µm]
marge_preDescemet = 30      # [µm] pre-Descemet membrane z-margin [default 30 µm]

save = True                # save figures and .txt/.csv files (bool)  
corr = True                 # compute posterior artefact correction (bool)
show = False               # display additional figures (bool)
user_validation = True     # ask user to validate segmentation results (bool)

#%% Preprocessing
from preproc_functions_def import preprocessing_OCT
import matplotlib.pyplot as plt

ProcessedImage, mask, coord_ROI, time_exe = preprocessing_OCT(filename, pathname, champ_analyse_mm, marge_postBowman, marge_preDescemet, save, corr, user_validation)

"""
OUTPUT :
    ProcessedImage -- image aplatie et corrigées pour les artefacts
    mask -- masque de correction pour l'artefact du stroma postérieur, dans la zone correspondant à la zone d'intérêt (ROI) analysée
    coord_ROI -- ((limites en x),(limites en z)) du stroma analysé sur l'image ProcessedImage
    time_exe -- temps d'exécution
"""

if (0):
    plt.close('all')
