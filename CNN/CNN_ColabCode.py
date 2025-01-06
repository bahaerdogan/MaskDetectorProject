import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from google.colab import drive
drive.mount('/content/drive')
train_dir = '/content/drive/MyDrive/CNNDataset/train'
val_dir = '/content/drive/MyDrive/CNNDataset/val'
train_gen = ImageDataGenerator(rescale=1.0/255, rotation_range=20, zoom_range=0.2, horizontal_flip=True)
val_gen = ImageDataGenerator(rescale=1.0/255)

train_data = train_gen.flow_from_directory(
    train_dir,
    target_size=(128, 128),  
    batch_size=32,
    class_mode='categorical'  
)

val_data = val_gen.flow_from_directory(
    val_dir,
    target_size=(128, 128),
    batch_size=32,
    class_mode='categorical'
)

model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 3)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(train_data.num_classes, activation='softmax')  # Sınıf sayısına göre çıktı katmanı
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])


history = model.fit(
    train_data,
    validation_data=val_data,
    epochs=80  
)


loss, accuracy = model.evaluate(val_data)
print(f"Validation Accuracy: {accuracy * 100:.2f}%")


model.save('/content/drive/My Drive/CNN_Model.h5')  # Modeli Google Drive'a kaydeder


from tensorflow.keras.preprocessing import image
import numpy as np

img_path = '/content/drive/MyDrive/CVdata/images/test/imagesa/ExternDisk0_ch4_20210810125301_20210810125400_37.jpg'
img = image.load_img(img_path, target_size=(128, 128))
img_array = image.img_to_array(img) / 255.0
img_array = np.expand_dims(img_array, axis=0)

prediction = model.predict(img_array)
class_indices = {v: k for k, v in train_data.class_indices.items()}  # Sınıfları ters çevirir
predicted_class = class_indices[np.argmax(prediction)]
print(f"Predicted Class: {predicted_class}")



import matplotlib.pyplot as plt

plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()




import locale
locale.getpreferredencoding = lambda: "UTF-8"
