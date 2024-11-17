import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout


# делим датасет на обучающую и тестовую выборку
# Сам датасет разделен на папку с фотографиями 28х28 (Х) и метками классов (У)
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

# Названия классов/меток = название одежды
classes_names = ['T-shirt/top', 'Trousers', 'Pullover', 'Dress',
                 'Coat', 'Sandal', 'Shirt', 'Sneakers', 'Bag', 'Ankle boot']

# нормализация данных
# нужна для того чтобы данные поступали на вход в диапозоне 0-1
# а не 0 - 255 (градация серого на кождой фотографии - интенсивность пикселей)
x_train = x_train / 255
x_test = x_test / 255

# Создание модели нейронной сети
model = Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),
    # Преобразует двумерный массив (со занчениями интенсивности каждого пикселя) в одномерный
    Dense(128, activation='relu'),  # определение количества нейронов на скрытом слое
    Dense(10, activation='softmax')  # выходной слой содержащий 10 нейронов по количеству классов

])

# Компиляция модели (указываем параметры обучения)
model.compile(optimizer=keras.optimizers.SGD(),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Напечатаем параметры модели
print(model.summary())

# Приступаем к обучению с учителем
print('Начинаем обучение модели')
model.fit(x_train, y_train, epochs=10)

# Проверка точности
print('\nПроверяем точность')
test_loss, teast_acc = model.evaluate(x_test, y_test)
print('Результат проверки - ', (teast_acc * 100) - ((teast_acc * 100) % 0.01), '%')

# Сохраняем модель в папку с программой !!!!!!!!!!!!!!!!!!!
model.save("./ready_model.h5")

# Предсказываем результат по новой картинке
prediction = model.predict(x_train)
test_photo_idx = 12
print(f'Предпологаемое занчение - {np.argmax(prediction[test_photo_idx])} \n Настоящий ответ - {y_train[test_photo_idx]}')

# Выводим изображение
plt.figure()
plt.imshow(x_train[test_photo_idx])
plt.colorbar()
plt.grid(False)
plt.show()
print(classes_names[np.argmax(prediction[test_photo_idx])])


