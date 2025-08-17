import torch
import torch.nn as nn
import torch.optim as optim

# Определяем архитектуру сети
class XORNet(nn.Module):
    def __init__(self):
        super(XORNet, self).__init__()
        self.fc1 = nn.Linear(2, 2)  # Входной слой (2 входа) -> Скрытый слой (2 нейрона)
        self.fc2 = nn.Linear(2, 1)  # Скрытый слой -> Выходной слой (1 нейрон)
        self.sigmoid = nn.Sigmoid()  # Функция активации
    
    def forward(self, x):
        x = torch.sigmoid(self.fc1(x))  # Активация скрытого слоя
        x = torch.sigmoid(self.fc2(x))  # Активация выходного слоя
        return x

# Создаем модель
model = XORNet()

# Определяем функцию потерь и оптимизатор
criterion = nn.BCELoss()  # Binary Cross Entropy
optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9)

# Подготавливаем данные для обучения
X = torch.tensor([[0., 0.], [0., 1.], [1., 0.], [1., 1.]])
y = torch.tensor([[0.], [1.], [1.], [0.]])

# Цикл обучения
epochs = 10000
for epoch in range(epochs):
    # Прямой проход
    outputs = model(X)
    loss = criterion(outputs, y)
    
    # Обратное распространение и оптимизация
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    # Вывод прогресса
    if (epoch+1) % 1000 == 0:
        print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.6f}')

# Тестируем модель
with torch.no_grad():
    predictions = model(X)
    predicted_labels = (predictions > 0.5).float()
    print("\nТестовые результаты:")
    print("Входные данные:", X.numpy())
    print("Предсказанные значения:", predicted_labels.numpy().flatten())
    print("Фактические значения:", y.numpy().flatten())
