import torch
from torch import nn

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.conv_stack_post = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=8, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Conv1d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Conv1d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Conv1d(64, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Dropout(0.2)
        )

        self.conv_stack_prompt = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=8, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Conv1d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Conv1d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Conv1d(64, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Dropout(0.2)
        )

        self.linear_relu_stack = nn.Sequential(
            nn.Linear(1027, 1024, bias=False),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(1024, 1024, bias=False),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(1024, 512, bias=False),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 256, bias=False),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 256, bias=False),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 256, bias=False),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 10, bias=False),
            nn.BatchNorm1d(10),
            nn.ReLU(),
            nn.Linear(10, 1)
        )


    def forward(self, x):
        # take the first 2048 values for the embeddings
        prompt_embedding = x[:, :1024]
        post_embedding = x[:, 1024:2048]
        features = x[:, 2048:]
        # take the rest of the values for the features
        prompt_embedding = prompt_embedding.unsqueeze(1)
        post_embedding = post_embedding.unsqueeze(1)

        prompt_embedding = self.conv_stack_prompt(prompt_embedding)
        post_embedding = self.conv_stack_post(post_embedding)

        prompt_embedding = self.flatten(prompt_embedding)
        post_embedding = self.flatten(post_embedding)

        # embeddings = self.flatten(embeddings)
        x = torch.cat([prompt_embedding, post_embedding, features], dim=1)
        # print(x.shape)
        # x = x.view(x.size(0), -1)
        logits = self.linear_relu_stack(x)
        return logits

def init_weights(m):
    if isinstance(m, nn.Conv1d) or isinstance(m, nn.Linear):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        if m.bias is not None:
            nn.init.zeros_(m.bias)

def get_model():
    model = NeuralNetwork().to(device)
    model.apply(init_weights)
    return model
