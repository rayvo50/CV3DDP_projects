import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
    
class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()

        self.encoder = nn.Sequential(
            nn.Linear(28*28,128),
            nn.ReLU(),
        )
        
        self.decoder = nn.Sequential(
            nn.Linear(128, 28*28),
            nn.ReLU(),
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


class SSIMLoss(torch.nn.Module):
    def __init__(self, window_size=11, size_average=True, channel=1, height=28, width=28):
        super(SSIMLoss, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.channel = channel
        self.height = height
        self.width = width

    def forward(self, x, y):
        x = x.view(-1, self.channel, self.height, self.width)
        y = y.view(-1, self.channel, self.height, self.width)        
        return 1 - self.ssim(x, y, self.window_size, self.size_average)

    def ssim(self, x, y, window_size, size_average, C1=0.01**2, C2=0.03**2):
        # use convolution to aply averaging over a window
        mu_x = F.conv2d(x, torch.ones((self.channel, 1, window_size, window_size)).to(x.device) / window_size**2, groups=self.channel)
        mu_y = F.conv2d(y, torch.ones((self.channel, 1, window_size, window_size)).to(y.device) / window_size**2, groups=self.channel)

        mu_x_sq = mu_x.pow(2)
        mu_y_sq = mu_y.pow(2)
        mu_x_mu_y = mu_x * mu_y

        sigma_x_sq = F.conv2d(x * x, torch.ones((self.channel, 1, window_size, window_size)).to(x.device) / window_size**2, groups=self.channel) - mu_x_sq
        sigma_y_sq = F.conv2d(y * y, torch.ones((self.channel, 1, window_size, window_size)).to(y.device) / window_size**2, groups=self.channel) - mu_y_sq
        sigma_xy = F.conv2d(x * y, torch.ones((self.channel, 1, window_size, window_size)).to(x.device) / window_size**2, groups=self.channel) - mu_x_mu_y

        ssim_map = ((2 * mu_x_mu_y + C1) * (2 * sigma_xy + C2)) / ((mu_x_sq + mu_y_sq + C1) * (sigma_x_sq + sigma_y_sq + C2))

        if size_average:
            return ssim_map.mean()
        else:
            return ssim_map.mean(1).mean(1).mean(1)




def train_autoencoder(model, data_loader, criterion, optimizer, val_loader, epochs=10):
    model.train()  # Set the model to training mode
    for epoch in range(epochs):
        running_loss = 0.0
        for data in data_loader:
            imgs, _ = data  # Labels are not needed
            imgs = imgs.view(imgs.size(0), -1)  # Flatten the images

            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, imgs)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        avg_loss = running_loss / len(data_loader)
        
        model.eval()  # Set model to evaluation mode
        validation_loss = 0.0
        with torch.no_grad():  # No gradients needed for validation
            for data in val_loader:
                inputs, _ = data
                outputs = model(inputs.view(inputs.size(0), -1))
                loss = criterion(outputs, inputs.view(inputs.size(0), -1))
                validation_loss += loss.item() * inputs.size(0)
        val_loss = validation_loss / len(val_loader)
    
        print(f'Epoch {epoch+1}/{epochs}, Train Loss: {avg_loss:.4f}, Validation Loss: {val_loss:.4f}')

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
    testset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=1)
    # Create a noisy dataset
    noisy_testset = []
    for i in range(int(0.1 * len(testset))):
        img, label = testset[i]
        noisy_testset.append((noisify(img, "salt_pepper"), label))
    noisy_test_loader = torch.utils.data.DataLoader(noisy_testset, batch_size=1)

    #display images
   # ex1_show_imgs(testset, noisy_testset)
    
    # load train dataset
    trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)

    # for corrupted trainset
    noisy_trainset = []
    for i in range(int(0.1 * len(trainset))):
        img, label = trainset[i]
        noisy_trainset.append((noisify(img, 'salt_pepper'), label))
    noisy_trainset = torch.utils.data.ConcatDataset([noisy_trainset, trainset])
    noisy_train_loader = torch.utils.data.DataLoader(noisy_trainset, batch_size=64, shuffle=True)

    # create model
    autoencoder = Autoencoder()
    loss_fn = SSIMLoss()
    optimizer = torch.optim.SGD(autoencoder.parameters(), lr=0.01, momentum=0.99)

    # train autoencoderdone
    train_autoencoder(autoencoder, noisy_train_loader, loss_fn, optimizer, test_loader, epochs=10)

    # evaluate perfromcande
    test_images, noisy_test_images, reconstructed, noisy_reconstructed = evaluate_autoencoder(autoencoder, test_loader, noisy_test_loader)

    # Plotting the images
    plot_images(test_images, noisy_test_images, reconstructed, noisy_reconstructed)




if __name__ == "__main__":
    main()