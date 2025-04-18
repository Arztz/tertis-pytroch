import torch
import torch.nn as nn
import torch.nn.functional as F

class HybridCNN(nn.Module):
    def __init__(self, action_dim, extra_input_dim, enable_dueling_dqn=True):
        super(HybridCNN, self).__init__()
        self.enable_dueling_dqn = enable_dueling_dqn

        # CNN สำหรับ field (1x20x10)
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),  # Output: (32, 20, 10)
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),  # Output: (64, 20, 10)
            nn.ReLU(),
            nn.Flatten()
        )

        # output จาก CNN: 64*20*10 = 12800
        cnn_output_dim = 64 * 20 * 10

        # fully connected จาก CNN + extra info
        total_input_dim = cnn_output_dim + extra_input_dim

        self.fc1 = nn.Linear(total_input_dim, 512)

        if self.enable_dueling_dqn:
            self.fc_value = nn.Linear(512, 256)
            self.value = nn.Linear(256, 1)

            self.fc_advantages = nn.Linear(512, 256)
            self.advantages = nn.Linear(256, action_dim)
        else:
            self.output = nn.Linear(512, action_dim)

    def forward(self, field_image, extra_vector):  # field_image: (batch, 1, 20, 10), extra_vector: (batch, n)
        x = self.cnn(field_image)  # → (batch, 12800)
        x = torch.cat([x, extra_vector.squeeze(1)], dim=1)  # รวมข้อมูลภาพ + ข้อมูลอื่นๆ

        x = F.relu(self.fc1(x))

        if self.enable_dueling_dqn:
            v = F.relu(self.fc_value(x))
            V = self.value(v)

            a = F.relu(self.fc_advantages(x))
            A = self.advantages(a)

            Q = V + A - A.mean(dim=1, keepdim=True)
        else:
            Q = self.output(x)
        return Q