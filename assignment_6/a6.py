import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader
from torch.nn import CrossEntropyLoss
from torch.optim import SGD
from torchvision import datasets
from torchvision.transforms import ToTensor
from tqdm import tqdm

training_data = datasets.MNIST(
    root="./data",
    train=True,
    download=True,
    transform=ToTensor()
)

test_data = datasets.MNIST(
    root="./data",
    train=False,
    download=True,
    transform=ToTensor()
)

train_dataloader = DataLoader(training_data, batch_size=64, shuffle=True)
test_dataloader = DataLoader(test_data, batch_size=64, shuffle=True)


train_features, train_labels = next(iter(train_dataloader))
img = train_features[0].squeeze()
label = train_labels[0]
plt.imshow(img, cmap="gray")
plt.show()

test_features, test_labels = next(iter(test_dataloader))
img = test_features[0].squeeze()
label = test_labels[0]
plt.imshow(img, cmap="gray")
plt.show()


L1_NEURONS = 200

class OneLayer(nn.Module):
    def __init__(self):
        super(OneLayer, self).__init__()

        self.fc1 = nn.Linear(28 * 28, L1_NEURONS)
        self.fc2 = nn.Linear(L1_NEURONS, 10)

    def forward(self, x):
        x = F.relu(self.fc1(x));
        x = F.relu(self.fc2(x))

        return x
    
model = OneLayer()
loss_fn = CrossEntropyLoss()
optimizer = SGD(model.parameters())


epochs = 10
running_loss = 0

for e in range(epochs):
    for i, data in tqdm(enumerate(train_dataloader)):
        model.train()

        inputs, labels = data
    
        optimizer.zero_grad()
        
        flattened_inputs = inputs.view(inputs.size(0), -1)
        outputs = model(flattened_inputs)

        loss = loss_fn(outputs, labels)
        loss.backward()

        optimizer.step()
        
        running_loss += loss.item()

    print(f"Epoch {e}: Average loss of {running_loss / 64} per batch")
    running_loss = 0

    # test benchmark
    model.eval()

    total_correct = 0
    total_samples = 0

    for i, (inputs, labels) in tqdm(enumerate(test_dataloader)):
        flattened_inputs = inputs.view(inputs.size(0), -1)

        outputs = model(flattened_inputs)
        _, predicted = torch.max(outputs, 1)

        total_correct += (predicted == labels).sum().item()
        total_samples += labels.size(0)

    accuracy = 100 * total_correct / total_samples
    print(f"Epoch {e}, Accuracy: {accuracy}")

torch.save(model.state_dict(), "model/one_layer_model.pth")


L1_NEURONS = 500
L2_NEURONS = 300

class TwoLayers(nn.Module):
    def __init__(self):
        super(TwoLayers, self).__init__()

        self.fc1 = nn.Linear(28 * 28, L1_NEURONS)
        self.fc2 = nn.Linear(L1_NEURONS, L2_NEURONS)
        self.fc3 = nn.Linear(L2_NEURONS, 10)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))

        return x
    
model = TwoLayers()
loss_fn = CrossEntropyLoss()
optimizer = SGD(model.parameters(), weight_decay=1e-4, lr=0.01) # weight decay for L2 regularization

epochs = 60
running_loss = 0

for e in range(epochs):
    for i, data in tqdm(enumerate(train_dataloader)):
        model.train()

        inputs, labels = data
    
        optimizer.zero_grad()
        
        flattened_inputs = inputs.view(inputs.size(0), -1)
        outputs = model(flattened_inputs)

        loss = loss_fn(outputs, labels)
        loss.backward()

        optimizer.step()
        
        running_loss += loss.item()

    print(f"Epoch {e}: Average loss of {running_loss / 64} per batch")
    running_loss = 0

    # test benchmark
    model.eval()

    total_correct = 0
    total_samples = 0

    for i, (inputs, labels) in tqdm(enumerate(test_dataloader)):
        flattened_inputs = inputs.view(inputs.size(0), -1)

        outputs = model(flattened_inputs)
        _, predicted = torch.max(outputs, 1)

        total_correct += (predicted == labels).sum().item()
        total_samples += labels.size(0)

    accuracy = 100 * total_correct / total_samples
    print(f"Epoch {e}, Accuracy: {accuracy}")

torch.save(model.state_dict(), "model/2-layer-0-01-learning-rate.pth")


L1_OUTPUT_CHANNELS = 16
L2_OUTPUT_CHANNELS = 32

class ConvNetwork(nn.Module):
    def __init__(self):
        super(ConvNetwork, self).__init__()

        # convolutional layers
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=L1_OUTPUT_CHANNELS, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=L1_OUTPUT_CHANNELS, out_channels=L2_OUTPUT_CHANNELS, kernel_size=3, padding=1)

        # pooling layer
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # fully-connected layers
        self.fc1 = nn.Linear(L2_OUTPUT_CHANNELS * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)


    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, L2_OUTPUT_CHANNELS * 7 * 7) # flatten images
        x = F.relu(self.fc1(x))
        x = self.fc2(x)

        return x

model = ConvNetwork()
loss_fn = CrossEntropyLoss()
optimizer = SGD(model.parameters(), weight_decay=1e-5, lr=0.01) # weight decay for L2 regularization


epochs = 60
running_loss = 0

for e in range(epochs):
    for i, data in tqdm(enumerate(train_dataloader)):
        model.train()

        inputs, labels = data
    
        optimizer.zero_grad()
        
        outputs = model(inputs)

        loss = loss_fn(outputs, labels)
        loss.backward()

        optimizer.step()
        
        running_loss += loss.item()

    print(f"Epoch {e}: Average loss of {running_loss / 64} per batch")
    running_loss = 0

    # test benchmark
    model.eval()

    total_correct = 0
    total_samples = 0

    for i, (inputs, labels) in tqdm(enumerate(test_dataloader)):
        outputs = model(inputs)
        _, predicted = torch.max(outputs, 1)

        total_correct += (predicted == labels).sum().item()
        total_samples += labels.size(0)

    accuracy = 100 * total_correct / total_samples
    print(f"Epoch {e}, Accuracy: {accuracy}")

torch.save(model.state_dict(), "model/conv-network.pth")