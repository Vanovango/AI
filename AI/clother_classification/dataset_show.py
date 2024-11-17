from tensorflow.keras.datasets import fashion_mnist
import matplotlib.pyplot as plt

(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
test_photo = x_train[12345]

plt.figure()
plt.imshow(test_photo)
plt.colorbar()
plt.grid(False)
plt.show()

