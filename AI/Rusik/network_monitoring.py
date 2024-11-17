# links
# https://pytorch.org/tutorials/beginner/basics/quickstart_tutorial.html
# we can bild network based on this tutorial

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torch.nn.functional as F

# Генерация случайных данных для обучения
data_size = 1000
input_size = 5
hidden_size = 32
output_size = 1

X = np.random.rand(data_size, input_size).astype(np.float32)
y = np.random.rand(data_size, output_size).astype(np.float32)


# Определение нейронной сети
class Network(nn.Module):
    def __init__(self):
        super(Network, self).__init__()
        self.input_layer = nn.Linear(input_size, hidden_size)
        self.output_layer = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = F.relu(self.input_layer(x))
        x = self.output_layer(x)
        return x


# Инициализация нейронной сети, функции потерь и оптимизатора
model = Network()

criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

min_loss = 10000

# Обучение нейронной сети
num_epochs = 100
for epoch in range(num_epochs):
    inputs = torch.from_numpy(X)
    targets = torch.from_numpy(y)

    optimizer.zero_grad()
    outputs = model(inputs)

    loss = criterion(outputs, targets)
    loss.backward()
    optimizer.step()

    if min_loss > loss:
        min_loss = loss
        # Сохранение модели
        torch.save(model.state_dict(), "./model.pth")

    print('Epoch [{}/{}], Loss: {:.4f}'.format(epoch + 1, num_epochs, loss.item()))

# Использование обученной нейронной сети для мониторинга
test_data = np.random.rand(1, input_size).astype(np.float32)
input_data = torch.from_numpy(test_data)
output_data = model(input_data)

print('Input data:', test_data)
print('Output prediction:', output_data.detach().numpy())
