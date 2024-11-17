import tensorflow as tf
from tensorflow.keras import models
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# Подгрузка готовой модели нейронной сети
model = models.load_model('./ready_model.h5')

# OPEN TEST PHOTO
image = Image.open('./cropped (3) (1).jpg').convert('L')
image_array = np.asarray(image)  # Преобразуем изображение в массив numpy
image_reshaped = np.reshape(image_array, (28, 28))  # Преобразуем массив в квадрат 28x28         784
image_reshaped = 255 - image_reshaped

# Тестовый принт с информацией о image_reshaped
# print(image_reshaped.shape, image_reshaped.dtype, image_reshaped)
    
# Выполняем предскозание
result_of_predict = model.predict(tf.expand_dims(image_reshaped, axis=0))

classes_names = ['T-shirt/top', 'Trousers', 'Pullover', 'Dress',
                 'Coat', 'Sandal', 'Shirt', 'Sneakers', 'Bag', 'Ankle boot']

print(result_of_predict)
print(f'Предпологаемое занчение - {classes_names[np.argmax(result_of_predict)]}')

# Выводим изображение
plt.figure()
plt.imshow(image)
plt.colorbar()
plt.text(x=0, y=10, s='Модель не првильно определяет объект', color='red', size=12)
plt.grid(False)
plt.show()
