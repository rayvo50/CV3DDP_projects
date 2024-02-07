import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
    
class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()

        self.encoder = nn.Sequential(
            nn.Linear(28*28, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(True),
            nn.Dropout(0.5),
            nn.Linear(256,128),
            nn.BatchNorm1d(128),
            nn.ReLU(True),
            # nn.Dropout(0.5),
            # nn.Linear(128, 64),
            # nn.BatchNorm1d(64),
            # nn.ReLU(True),
            # #nn.Linear(64, 32)  # Latent space representation
        )
        
        # Decoder
        self.decoder = nn.Sequential(
            # nn.Linear(32, 64),
            # nn.BatchNorm1d(64),
            # nn.ReLU(True),
            # nn.Dropout(0.2),
            # nn.Linear(64, 128),
            # nn.BatchNorm1d(128),
            # nn.ReLU(True),
            # nn.Dropout(0.5),
            nn.Linear(128, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(True),
            nn.Linear(256, 28*28),
            nn.Sigmoid()  # Ensures output is in the range [0, 1]
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

def train_autoencoder(model, train_loader, criterion, optimizer, val_loader, epochs=10):
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for data in train_loader:
            imgs, _ = data  # Labels are not needed
            imgs = imgs.view(imgs.size(0), -1)  # Flatten the images

            optimizer.zero_grad()
            outputs = model(imgs)
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
                outputs = model(inputs.view(inputs.size(0), -1))
                loss = criterion(outputs, inputs.view(inputs.size(0), -1))
                validation_loss += loss.item() * inputs.size(0)
        val_loss = validation_loss / len(val_loader)
    
        print(f'Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}')

def evaluate_autoencoder(model, test_loader, noisy_test_loader):
    model.eval()  # Set the model to evaluation mode
    original_images, noisy_images, reconstructed, noisy_reconstructed = [], [], [], []

    with torch.no_grad():
        # Process original images
        for imgs, _ in test_loader:
            imgs = imgs.view(imgs.size(0), -1)
            outputs = model(imgs)
            original_images.append(imgs)
            reconstructed.append(outputs)

        # Process noisy images
        for imgs, _ in noisy_test_loader:
            imgs = imgs.view(imgs.size(0), -1)
            outputs = model(imgs)
            noisy_images.append(imgs)
            noisy_reconstructed.append(outputs)

    return original_images, noisy_images, reconstructed, noisy_reconstructed
   

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

    # create model
    autoencoder = Autoencoder()
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(autoencoder.parameters(), lr=0.001)

    # train autoencoder
    train_autoencoder(autoencoder, train_loader, loss_fn, optimizer, test_loader, epochs=5)

    # evaluate perfromcande
    test_images, noisy_test_images, reconstructed, noisy_reconstructed = evaluate_autoencoder(autoencoder, test_loader, noisy_test_loader)

    # Plotting the images
    plot_images(test_images, noisy_test_images, reconstructed, noisy_reconstructed)



  

if __name__ == "__main__":
    main()