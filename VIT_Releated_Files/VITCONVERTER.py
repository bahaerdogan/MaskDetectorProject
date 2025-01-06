import os
import cv2
from shutil import copyfile

yolo_images_path = r"D:\MyWork\ComputerVisionMaskDetector\ComputerVisionMaskDetector\MaskDetectorData\EnlargedDataSet\Train\images"  # YOLO görüntüleri
yolo_labels_path = r"D:\MyWork\ComputerVisionMaskDetector\ComputerVisionMaskDetector\MaskDetectorData\EnlargedDataSet\Train\labels"  # YOLO etiketleri
vit_dataset_path = r"D:\MyWork\ComputerVisionMaskDetector\ComputerVisionMaskDetector\MaskDetectorData\VITformat"          # ViT formatındaki dataset


class_names = ["with_mask", "without_mask", "mask_weared_incorrect"]

train_images_path = os.path.join(vit_dataset_path, "train")
os.makedirs(train_images_path, exist_ok=True)

for label_file in os.listdir(yolo_labels_path):
    label_path = os.path.join(yolo_labels_path, label_file)
    image_path = os.path.join(yolo_images_path, label_file.replace(".txt", ".jpg"))
    
    if not os.path.exists(image_path):
        continue
    
    with open(label_path, "r") as f:
        lines = f.readlines()
    
    if lines:
        class_id = int(lines[0].split()[0])
        class_name = class_names[class_id]
        
        class_folder = os.path.join(train_images_path, class_name)
        os.makedirs(class_folder, exist_ok=True)
        
        img = cv2.imread(image_path)
        img_resized = cv2.resize(img, (224, 224))  
        output_image_path = os.path.join(class_folder, os.path.basename(image_path))
        cv2.imwrite(output_image_path, img_resized)

print("Dönüşüm tamamlandı. Veriler ./vit_dataset/train dizininde saklandı.")
