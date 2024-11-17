from tensorflow.keras import models
from tensorflow.keras.datasets import fashion_mnist
import numpy as np
import matplotlib.pyplot as plt


model = models.load_model('./ready_model.h5')

(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
classes_names = ['T-shirt/top', 'Trousers', 'Pullover', 'Dress',
                 'Coat', 'Sandal', 'Shirt', 'Sneakers', 'Bag', 'Ankle boot']

test_photo_idx = 40

print(x_train[test_photo_idx].shape, x_train[test_photo_idx].dtype, x_train[test_photo_idx])
# Предсказываем результат по картинке из датасета
prediction = model.predict(x_train)
#print(x_train[test_photo_idx].shape, x_train[test_photo_idx])
print(f'Предпологаемое занчение - {np.argmax(prediction[test_photo_idx])} \n Настоящий ответ - {y_train[test_photo_idx]}')
print(classes_names[np.argmax(prediction[test_photo_idx])])

# Выводим изображение
plt.figure()
plt.imshow(x_train[test_photo_idx])
plt.colorbar()
plt.grid(False)
plt.show()
