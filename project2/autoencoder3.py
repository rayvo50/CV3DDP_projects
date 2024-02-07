import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import random
import seaborn as sns
from sklearn.metrics import confusion_matrix

    
class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
        
        # Encoder
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2, return_indices=True)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        
        # Decoder
        self.unpool = nn.MaxUnpool2d(2, 2)
        self.deconv1 = nn.ConvTranspose2d(64, 32, 3, padding=1)
        self.deconv2 = nn.ConvTranspose2d(32, 1, 3, padding=1)

        self.history = {'loss': [], 'val_loss': []}
        self.threshold = 0.03412
        
    def forward(self, x):
        # Encoder
        x = F.relu(self.conv1(x))
        x, indices1 = self.pool(x)
        x = F.relu(self.conv2(x))
        size2 = x.size()
        x, indices2 = self.pool(x)
        
        # Decoder
        x = self.unpool(x, indices2, output_size=size2)
        x = F.relu(self.deconv1(x))
        x = self.unpool(x, indices1)
        x = torch.sigmoid(self.deconv2(x))
        return x
    

def train_autoencoder(model, train_loader, criterion, optimizer, val_loader, epochs=10):
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for data in train_loader:
            imgs, _ = data  # Labels are not needed

            optimizer.zero_grad()
            outputs = model(augment(imgs))
            loss = criterion(outputs, imgs)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
        
        train_loss = running_loss / len(train_loader)
        
        model.eval()  # Set model to evaluation mode
        validation_loss = 0.0
        with torch.no_grad():  # No gradients needed for validation
            for data in val_loader:
                inputs, _ = data
                outputs = model(inputs)
                loss = criterion(outputs, inputs)
                validation_loss += loss.item() * inputs.size(0)
        val_loss = validation_loss / len(val_loader)
        
        model.history['loss'].append(train_loss)
        model.history['val_loss'].append(val_loss)
        print(f'Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}')

def evaluate_autoencoder(model, test_loader, noisy_test_loader, criterion):
    model.eval()  # Set the model to evaluation mode
    original_images, noisy_images, reconstructed, noisy_reconstructed = [], [], [], []
    lossess_original = []
    lossess_noisy = []
    predictions = []
    with torch.no_grad():
        # Process original images
        for imgs, _ in test_loader:
            outputs = model(imgs)
            original_images.append(imgs)
            reconstructed.append(outputs)
            loss = criterion(outputs, imgs).tolist()
            lossess_original.append(loss)
            if loss < model.threshold:
                predictions.append(0)
            else:     
                predictions.append(1)
            
        # Process noisy images
        for imgs, _ in noisy_test_loader:
            outputs = model(imgs)
            noisy_images.append(imgs)
            noisy_reconstructed.append(outputs)
            loss = criterion(outputs, imgs).tolist()
            lossess_noisy.append(loss)
            if loss < model.threshold:
                predictions.append(0)
            else:     
                predictions.append(1)

    return original_images, noisy_images, reconstructed, noisy_reconstructed, lossess_original, lossess_noisy, predictions
   

# Function to add noise to images
def noisify(img, noise_type, noise_level=0.4):
    noisy_img = img.clone() 
    if noise_type == 'gaussian':
        noisy_img = noisy_img  + torch.randn(*noisy_img.shape)*noise_level
    if noise_type == 'salt_pepper':   
        num_noise_pixels = int(noise_level * img.numel())  # number of pixels to noisifyS
        noisy_img.view(-1)[torch.randperm(img.numel())[:num_noise_pixels // 2].tolist()] = 1    # salt
        noisy_img.view(-1)[torch.randperm(img.numel())[:num_noise_pixels // 2].tolist()] = 0    # pepper
    
    return torch.clamp(noisy_img, 0.0, 1.0)

def augment(img):
    p = random.random()
    if 0 <= p < 0.6:
        return img
    elif 0.6 <= p < 0.8:
        return noisify(img, "salt_pepper")
    elif 0.8 <= p < 1:
        return noisify(img, "gaussian")
        

def ex1_show_imgs(testset, noisy_testset):
    # Visualizing original and corrupted images
    fig, axs = plt.subplots(2, 4, figsize=(10, 5))
    i = -1
    for (noisy_img, label) in noisy_testset:
        i+=1
        if i >= 4:
            break
        # Get the corresponding original image
        original_img, _ = testset[i]
        # Remove channel dimension for displaying
        original_img = original_img.squeeze()
        noisy_img = noisy_img.squeeze()
        # Original image
        axs[0, i].imshow(original_img.numpy(), cmap='gray')
        axs[0, i].set_title(f'Original - {label}')
        axs[0, i].axis('off')
        # Corrupted image
        axs[1, i].imshow(noisy_img.numpy(), cmap='gray')
        axs[1, i].set_title(f'Corrupted - {label}')
        axs[1, i].axis('off')

    plt.tight_layout()
    plt.show()


def plot_images(original, noisy, reconstructed, noisy_reconstructed):
    fig, axs = plt.subplots(2, 3)
    fig.subplots_adjust(top=0.85)  # Adjust top to make space for titles
    for i in range(3):  # Two rows
            axs[0, i].imshow(original[i].numpy().reshape(28, 28), cmap='gray')
            axs[0, i].axis('off')
            axs[1, i].imshow(reconstructed[i].numpy().reshape(28, 28), cmap='gray')
            axs[1, i].axis('off')
            
    for i, title in enumerate(["Original Images", "Reconstructed Images" ]):
        # Place titles in the center above each row
        fig.text(0.5, 0.94 if i == 0 else 0.47, title, ha='center', va='center', fontsize=16, fontweight='bold')
    
    fig2, axs2 = plt.subplots(2, 3)
    fig2.subplots_adjust(top=0.85)  # Adjust top to make space for titles
    for i in range(3):  # Two rows
            axs2[0, i].imshow(noisy[i].numpy().reshape(28, 28), cmap='gray')
            axs2[0, i].axis('off')
            axs2[1, i].imshow(noisy_reconstructed[i].numpy().reshape(28, 28), cmap='gray')
            axs2[1, i].axis('off')
            
    for i, title in enumerate(["Corrupted Images", "Reconstructed from Corrupted Images" ]):
        # Place titles in the center above each row
        fig2.text(0.5, 0.94 if i == 0 else 0.47, title, ha='center', va='center', fontsize=16, fontweight='bold')    
    plt.show()

def plot_loss(model):
    plt.figure()
    plt.plot(model.history['loss'], label='Training Loss')
    plt.plot(model.history['val_loss'], label='Test Loss')
    plt.title('Loss Over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid()
    plt.show()

def plot_tresh(losses_original, losses_noisy):
    sns.kdeplot(losses_original, bw_adjust=0.5, color="blue", label="Normal Images")
    sns.kdeplot(losses_noisy, bw_adjust=0.5, color="red", label="Corrupted Images")
    print(np.max(losses_original),np.min(losses_noisy))
    middle_point = (np.max(losses_original) + np.min(losses_noisy)) / 2
    plt.axvline(x=middle_point, color='green', linestyle='--', ymin=0, ymax=0.05, label=f"Threshold = {middle_point:.5f}")

    plt.legend()
    plt.title('Loss distribution')
    plt.xlabel('MSE error')
    plt.ylabel('Density')
    plt.show()

def plot_confusion_matrix(true_labels, predictions):
    cm = confusion_matrix(true_labels, predictions)
    plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=["Normal [0]","Anomaly [1]"], yticklabels=["Predicted [0]","Anomaly [1]"])
    plt.xlabel('Predicted labels')
    plt.ylabel('True labels')
    plt.title('Confusion Matrix')
    plt.show()


def main():

    transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    # load test dataset
    testset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transforms.ToTensor())
    test_loader = torch.utils.data.DataLoader(testset, batch_size=1)

    # Create a noisy dataset
    noisy_testset = []
    for i in range(int(0.1 * len(testset))):
        img, label = testset[i]
        noisy_testset.append((noisify(img, "salt_pepper"), label))
    noisy_test_loader = torch.utils.data.DataLoader(noisy_testset, batch_size=1)

    
    # load train dataset
    trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=50, shuffle=True)

    # create modelgaussian
    autoencoder = Autoencoder()
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(autoencoder.parameters(), lr=0.001)

    # train autoencoder
    train_autoencoder(autoencoder, train_loader, loss_fn, optimizer, test_loader, epochs=3)

    # evaluate performance
    test_images, noisy_test_images, reconstructed, noisy_reconstructed, losses_original, losses_noisy, predictions  = evaluate_autoencoder(autoencoder, test_loader, noisy_test_loader, loss_fn)
    # Plotting the images
    plot_images(test_images, noisy_test_images, reconstructed, noisy_reconstructed)
    # plot loss
    plot_loss(autoencoder)
    # plot treshold
    plot_tresh(losses_original, losses_noisy)
    true_labels = np.concatenate([np.zeros(len(testset)), np.ones(len(noisy_testset))])
    plot_confusion_matrix(true_labels, predictions)
    
    torch.save(autoencoder.state_dict(), "./autoencoder")

  

if __name__ == "__main__":
    main()