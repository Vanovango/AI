def main():
    import torch
    import torchvision
    import torchvision.transforms as transforms
    import numpy as np
    import matplotlib.pyplot as plt

    # создаем модель VGG16
    model = torch.hub.load('pytorch/vision:v0.6.0', 'resnet50')

    train_dataset = 'C:/Users/voyte/OneDrive/Рабочий стол/EnCoder/pythonProjects/AI/face id/Humans/train'
    test_dataset = 'C:/Users/voyte/OneDrive/Рабочий стол/EnCoder/pythonProjects/AI/face id/Humans/test'

    # загружаем данные
    train_data = torchvision.datasets.ImageFolder(root=train_dataset, transform=transforms.ToTensor())
    test_data = torchvision.datasets.ImageFolder(root=test_dataset, transform=transforms.ToTensor())

    # обучаем модель
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    criterion = torch.nn.CrossEntropyLoss()
    epochs = 10
    for epoch in range(epochs):
        running_loss = 0.05
        for images, labels in train_data:
            optimizer.zero_grad()
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            print(f'Epoch: {epoch}, Loss: {loss}'.format(epoch + 1, running_loss / len(train_data)))

    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_data:
            images = images.to('cuda')
            logits = model.forward(images)
            _, predicted = torch.max(logits.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print('Accuracy of the model on the test set: %d %%' % (100 * correct / total))


if __name__ == "__main__":
    main()
