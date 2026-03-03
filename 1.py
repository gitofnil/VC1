import cv2
import numpy as np
import os

base_dir = 'highway'
input_dir = os.path.join(base_dir, 'input')
gt_dir = os.path.join(base_dir, 'groundtruth')

start_idx = 1051
end_idx = 1350
train_size = 150

train_images = []
test_images = []
gt_images = []

print("1. Carregant imatges del dataset...")
for i in range(start_idx, end_idx + 1):
    img_name = f"in{i:06d}.jpg"
    img_path = os.path.join(input_dir, img_name)
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    
    if img is not None:
        if i < start_idx + train_size:
            train_images.append(img)
        else:
            test_images.append(img)
            gt_name = f"gt{i:06d}.png"
            gt_path = os.path.join(gt_dir, gt_name)
            gt_img = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)
            gt_images.append(gt_img)

train_images = np.array(train_images, dtype=np.float32)
test_images = np.array(test_images, dtype=np.float32)

print("2. Calculant fons (µ i σ)...")
mu = np.mean(train_images, axis=0)
sigma = np.std(train_images, axis=0)

height, width = mu.shape


print("3. Creant màscara geomètrica per tapar els arbres...")
roi_bin = np.zeros((height, width), dtype=np.uint8)
# Coordenades d'un polígon que només cobreix l'asfalt de la carretera
pts = np.array([[150, 0], [width, 0], [width, height], [20, height]], np.int32)
cv2.fillPoly(roi_bin, [pts], 1)


print("4. Iniciant segmentació i creació del vídeo...")
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out_video = cv2.VideoWriter('output_detections.mp4', fourcc, 15.0, (width, height), isColor=False)

alpha = 2.5 
beta = 10.0  

# Morfologia: Obrim una mica per treure soroll, tanquem molt per compactar el cotxe
kernel_open = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
kernel_close = cv2.getStructuringElement(cv2.MORPH_RECT, (11, 11)) 

acc_cas1_basic = []      
acc_cas2_elaborat = []   
acc_cas3_morfologia = [] 

for idx, test_img in enumerate(test_images):
    diff = np.abs(test_img - mu)
    
    # --- CAS 1: Model Bàsic ---
    mask_basic_raw = (diff > 30).astype(np.uint8) * 255
    mask_basic = cv2.bitwise_and(mask_basic_raw, mask_basic_raw, mask=roi_bin)
    
    # --- CAS 2: Model Elaborat ---
    threshold_matrix = (alpha * sigma) + beta
    mask_advanced_raw = (diff > threshold_matrix).astype(np.uint8) * 255
    mask_advanced = cv2.bitwise_and(mask_advanced_raw, mask_advanced_raw, mask=roi_bin)
    
    # --- CAS 3: Model Elaborat + Morfologia ---
    mask_clean = cv2.morphologyEx(mask_advanced, cv2.MORPH_OPEN, kernel_open)
    mask_clean = cv2.morphologyEx(mask_clean, cv2.MORPH_CLOSE, kernel_close)
    
    cv2.imshow("Video Final (Sense arbres)", mask_clean)
    out_video.write(mask_clean)
    
    if cv2.waitKey(30) & 0xFF == ord('q'):
        break
    
    # --- Avaluació ---
    gt = gt_images[idx]
    if gt is not None:
        gt_bin = (gt > 170).astype(np.uint8) 
        total_pixels_roi = np.sum(roi_bin)
        
        if total_pixels_roi > 0:
            pred_basic = (mask_basic > 0).astype(np.uint8)
            acc_cas1_basic.append(np.sum((gt_bin == pred_basic) & (roi_bin == 1)) / total_pixels_roi)
            
            pred_adv = (mask_advanced > 0).astype(np.uint8)
            acc_cas2_elaborat.append(np.sum((gt_bin == pred_adv) & (roi_bin == 1)) / total_pixels_roi)
            
            pred_clean = (mask_clean > 0).astype(np.uint8)
            acc_cas3_morfologia.append(np.sum((gt_bin == pred_clean) & (roi_bin == 1)) / total_pixels_roi)

out_video.release()
cv2.destroyAllWindows()

print("\n=== RESULTATS TASCA 6: AVALUACIÓ (ACCURACY) ===")
print(f"Cas 1 (Model Bàsic, thr=30)             : {np.mean(acc_cas1_basic):.4f}")
print(f"Cas 2 (Model Elaborat, alpha={alpha})       : {np.mean(acc_cas2_elaborat):.4f}")
print(f"Cas 3 (Model Elaborat + Morfologia)     : {np.mean(acc_cas3_morfologia):.4f}")
print("===============================================")