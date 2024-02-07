
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import Adam
from torchvision import datasets, transforms
import random
import matplotlib.pyplot as plt


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=5)
        self.bn1 = nn.BatchNorm2d(num_features=32)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3)
        self.bn2 = nn.BatchNorm2d(num_features=64)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout1 = nn.Dropout(0.2)
        
        self.fc1 = nn.Linear(in_features=1600, out_features=128)
        self.bn3 = nn.BatchNorm1d(num_features=128)
        self.dropout2 = nn.Dropout(0.5)

        self.fc2 = nn.Linear(in_features=128, out_features=128)
        self.bn4 = nn.BatchNorm1d(num_features=128)
        self.dropout3 = nn.Dropout(0.5)

        self.fc3 = nn.Linear(in_features=128, out_features=10)

        self.history = {'loss': [], 'accuracy': [], 'val_loss': [], 'val_accuracy': []}

    def forward(self, x):
        x = self.conv1(x) 
        x = self.bn1(x)
        x = F.relu(x)
        x = self.pool1(x)
        # ------------------------- #
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.pool2(x)
        x = self.dropout1(x)
        # ------------------------- #
        x = x.view(x.size(0), -1) 
        x = self.fc1(x)
        x = self.bn3(x)
        x = F.relu(x)
        # ------------------------- #
        x = self.fc2(x)
        x = self.bn4(x)
        x = F.relu(x)
        x = self.dropout3(x)
        # ------------------------- #
        x = self.fc3(x)
        x = F.log_softmax(x, dim=1)
        
        return x



# Function to train the model
def train_one_epoch(model, device, train_dataloader, loss_fn, optimizer):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for i, data in enumerate(train_dataloader):
        inputs, labels = data[0].to(device), data[1].to(device)
        # zero gradient for every batch
        optimizer.zero_grad()
        # make predictions
        outputs = model(inputs)
        # compute loss
        loss = loss_fn(outputs, labels)
        # back propagate
        loss.backward()
        # adjust weights
        optimizer.step()
        
        # save for computing loss and accuracy
        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    train_loss = running_loss / len(train_dataloader)
    train_accuracy = correct / total
    model.history['loss'].append(train_loss)
    model.history['accuracy'].append(train_accuracy)
    return train_loss, train_accuracy

# Function to evaluate the model
def evaluate(model, device, val_dataloader, loss_fn):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    # disable gradient computation
    with torch.no_grad():
        for i, data in enumerate(val_dataloader):
            inputs, labels = data[0].to(device), data[1].to(device)
            # make predictions
            outputs = model(inputs)
            # compute loss
            loss = loss_fn(outputs, labels)

            # save for computing loss and accuracy
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    test_loss = running_loss / len(val_dataloader)
    test_accuracy = correct / total
    model.history['val_loss'].append(test_loss)
    model.history['val_accuracy'].append(test_accuracy)
    return test_loss, test_accuracy

# Define a custom transform function for data augmentation
data_transforms = transforms.Compose([
        transforms.RandomRotation(10),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))  
    ])
def custom_transform(image):
    if random.random() < 0.05:
        return data_transforms(image)
    else:
        return transforms.Compose([transforms.ToTensor(),
                                   transforms.Normalize((0.1307,), (0.3081,))])(image)
    

def main():

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    # generate test dataset
    train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform) # = custom_transform for data augmentation
    test_dataset = datasets.MNIST(root='./data', train=False, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    # create model
    model = Model()
    loss_fn = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=0.001)

    # Prepare for training
    device = torch.device("cpu")
    model.to(device)

    # training and validation
    epochs = 100
    for epoch in range(epochs):
        train_loss, train_accuracy = train_one_epoch(model, device, train_loader, loss_fn, optimizer)
        val_loss, val_accuracy = evaluate(model, device, test_loader, loss_fn)
        print(f"Epoch: {epoch+1}/{epochs}  |  Loss: {train_loss:.4f}  |  Accuracy: {train_accuracy:.4f}  |  Val Loss: {val_loss:.4f}  |  Val Accuracy: {val_accuracy:.4f}")

    # plot loss and accurcy over epochs
    _, ax = plt.subplots(ncols=2, figsize=(15,5))
    ax[0].plot(model.history['loss'], label='Training Loss')
    ax[0].plot(model.history['val_loss'], label='Test Loss')
    ax[0].set_title('Loss Over Epochs')
    ax[0].set_xlabel('Epochs')
    ax[0].set_ylabel('Loss')
    ax[0].legend()
    ax[0].grid()
    ax[1].plot(model.history['accuracy'], label='Training Accuracy')
    ax[1].plot(model.history['val_accuracy'], label='Test Accuracy')
    ax[1].set_title('Accuracy Over Epochs')
    ax[1].set_xlabel('Epochs')
    ax[1].set_ylabel('Accuracy')
    ax[1].legend()
    ax[1].grid()
    plt.show()

if __name__ == "__main__":
    main()

