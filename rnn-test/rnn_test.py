import torch
import torch.nn as nn
import torch.nn.functional as F

class RecursiveNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(RecursiveNN, self).__init__()
        self.hidden_size = hidden_size
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size * 2, hidden_size)  # ����������� ����
        self.fc3 = nn.Linear(hidden_size, output_size)
        
    def forward(self, input1, input2):
        # ������ ����� ������ �������������� ����
        output1 = F.relu(self.fc1(input1))
        output2 = F.relu(self.fc1(input2))
        
        # ����������� �����������
        combined = torch.cat((output1, output2), dim=1)
        
        # ������ ����� ����������� ����
        recursive_output = F.relu(self.fc2(combined))
        
        # ������ ����� ������ �������������� ���� � ��������� ������������
        output = self.fc3(recursive_output)
        return output

# ������ �������������
input_size = 300  # ������ ���������� ������������� ������� �����
hidden_size = 100  # ������ �������� ����
output_size = 1  # �������� ������

model = RecursiveNN(input_size, hidden_size, output_size)

# ������ ������� ������
input1 = torch.randn(1, input_size)  # ������ ������� �����
input2 = torch.randn(1, input_size)  # ������ ������� �����
input3 = torch.randn(1, input_size)
input4 = torch.randn(1, input_size)
# ��������� ������������
output = model(input1, input2)
print(output)
