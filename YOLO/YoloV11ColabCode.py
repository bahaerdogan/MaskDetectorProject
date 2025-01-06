pip install ultralytics==8.3.53

from google.colab import drive
drive.mount('/content/drive')


!ls /content/drive/MyDrive/GenişletilmişVeriSeti


data_yaml = """
train: /content/drive/MyDrive/GenişletilmişVeriSeti/Train/images
val: /content/drive/MyDrive/GenişletilmişVeriSeti/Val/images

nc: 3
names:
  - with_mask
  - without_mask
  - mask_weared_incorrect
"""

with open('data.yaml', 'w') as f:
    f.write(data_yaml)

print("data.yaml updated successfully.")


from ultralytics import YOLO


model = YOLO("yolo11n.pt")

model.train(data='data.yaml', epochs=80, imgsz=640, batch=32)


import locale
locale.getpreferredencoding = lambda: "UTF-8"



!zip -r 'run.zip' /content/runs