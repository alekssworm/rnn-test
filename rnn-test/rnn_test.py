import torch
import torch.nn as nn
import torch.nn.functional as F

class RecursiveNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(RecursiveNN, self).__init__()
        self.hidden_size = hidden_size
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size * 2, hidden_size)  # Рекурсивный слой
        self.fc3 = nn.Linear(hidden_size, output_size)
        
    def forward(self, input1, input2):
        # Проход через первый полносвязанный слой
        output1 = F.relu(self.fc1(input1))
        output2 = F.relu(self.fc1(input2))
        
        # Объединение результатов
        combined = torch.cat((output1, output2), dim=1)
        
        # Проход через рекурсивный слой
        recursive_output = F.relu(self.fc2(combined))
        
        # Проход через второй полносвязанный слой и получение предсказания
        output = self.fc3(recursive_output)
        return output

# Пример использования
input_size = 300  # Размер векторного представления каждого слова
hidden_size = 100  # Размер скрытого слоя
output_size = 1  # Выходной размер

model = RecursiveNN(input_size, hidden_size, output_size)

# Пример входных данных
input1 = torch.randn(1, input_size)  # Первый входной текст
input2 = torch.randn(1, input_size)  # Второй входной текст
input3 = torch.randn(1, input_size)
input4 = torch.randn(1, input_size)
# Получение предсказания
output = model(input1, input2)
print(output)
