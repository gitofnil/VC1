import numpy as np
import pandas as pd
import cv2 
import os 


path = 3
imatges_test = []
imatges_train =  []
dp = os.listdir(path) #path de highway

for i in range(1051, 1351):

    filename = f"in{i:06d}.jpg" 
    full_path = os.path.join(path, filename)
    
    # Llegir en escala de grisos com demana la pràctica [cite: 38, 42]
    img = cv2.imread(full_path, cv2.IMREAD_GRAYSCALE)
    
    if img is not None:
        if i <= 1200:
            imatges_train.append(img)
        else:
            imatges_test.append(img)