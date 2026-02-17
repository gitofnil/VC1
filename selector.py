import numpy as np
import pandas as pd
import cv2 
import os 
import math 
import statistics

path = "../VC1/highway/input"
imatges_test = []
imatges_train =  []
dp = os.listdir(path) #path de highway

for i in range(1051, 1351):

    filename = f"in{i:06d}.jpg" #
    full_path = os.path.join(path, filename) #afegeix el nou path al path original
    
    img = cv2.imread(full_path, cv2.IMREAD_GRAYSCALE) # converteix la iamtge a grisos
    
    if img is not None:
        if i <= 1200:
            imatges_train.append(img)
        else:
            imatges_test.append(img)



matrix_train = np.array(imatges_train) # el passem la iamtge a un array
        
mitjana = np.mean(matrix_train, axis = 0) # axis = 0 mira el mateix pixel a totes les imatges 
desviacio = np.std(matrix_train, axis = 0)
    

for i in matrix_train.size()

if(mitjana)