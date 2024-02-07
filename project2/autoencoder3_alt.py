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
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(
                32, 32, kernel_size=3, stride=2, padding=1, output_padding=1
            ),
            nn.ReLU(),
            nn.ConvTranspose2d(
                32, 32, kernel_size=3, stride=2, padding=1, output_padding=1
            ),
            nn.ReLU(),
            nn.Conv2d(32, 1, kernel_size=3, padding=1),
            nn.Sigmoid(),
        )

        self.history = {'loss': [], 'val_loss': []}
        

    def forward(self, x: torch.Tensor):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded
    
    # -----------------------------------------------------------------------------------------------------------    
    #     self.encoder = nn.Sequential(
    #         nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1),
    #         nn.ReLU(),
    #         nn.MaxPool2d(kernel_size=2, stride=2),
    #         nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1),
    #         nn.ReLU(),
    #         nn.MaxPool2d(kernel_size=2, stride=2)
    #     )
    #     self.decoder = nn.Sequential(
    #         nn.ConvTranspose2d(16, 16, 
    #                            kernel_size=3, 
    #                            stride=2, 
    #                            padding=1, 
    #                            output_padding=1),
    #         nn.ReLU(),
    #         nn.ConvTranspose2d(16, 1, 
    #                            kernel_size=3, 
    #                            stride=2, 
    #                            padding=1, 
    #                            output_padding=1),
    #         nn.Sigmoid()
    #     )
    #     self.history = {'loss': [], 'val_loss': []}
         
    # def forward(self, x):
    #     x = self.encoder(x)
    #     x = self.decoder(x)
    #     return x
    # -----------------------------------------------------------------------------------------------------------    
    #     # Encoder
    #     self.pool = nn.MaxPool2d(2, 2, return_indices=True)
    #     self.conv1 = nn.Conv2d(1, 8, 3, padding=1)
    #     self.conv2 = nn.Conv2d(8, 16, 3, padding=1)
    #     self.conv3 = nn.Conv2d(8, 8, 3, padding=1)
    #     self.dropout = nn.Dropout(p=0.2)
        
        
    #     # Decoder
    #     self.unpool = nn.MaxUnpool2d(2, 2)
    #     self.deconv1 = nn.ConvTranspose2d(8, 8, 3, padding=1)
    #     self.deconv2 = nn.ConvTranspose2d(16, 8, 3, padding=1)
    #     self.deconv3 = nn.ConvTranspose2d(8, 1, 3, padding=1)

    #     self.history = {'loss': [], 'val_loss': []}
        
    # def forward(self, x):
    #     # Encoder
    #     x = F.relu(self.conv1(x))
    #     x = self.dropout(x)
    #     x, indices1 = self.pool(x)
    #     x = F.relu(self.conv2(x))
    #     x = self.dropout(x)
    #     size2 = x.size()
    #     x, indices2 = self.pool(x)
    #    # x = F.relu(self.conv3(x))
    #  #   size3 = x.size()
    #   #  x, indices3 = self.pool(x)
        
    #     # Decoder
    #    # x = self.unpool(x, indices3, output_size=size3)
    #     #x = F.relu(self.deconv1(x))
    #     #x = self.dropout(x)
    #     x = self.unpool(x, indices2, output_size=size2)
    #     x = F.relu(self.deconv2(x))
    #     x = self.dropout(x)
    #     x = self.unpool(x, indices1)
    #     x = torch.sigmoid(self.deconv3(x))
    #     return x
    


def train_autoencoder(model, train_loader, criterion, optimizer, val_loader, epochs=10):
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for data in train_loader:
            imgs, _ = data  # Labels are not needed

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
            lossess_original.append(criterion(outputs, imgs).tolist())
            
            
        # Process noisy images
        for imgs, _ in noisy_test_loader:
            outputs = model(imgs)
            noisy_images.append(imgs)
            noisy_reconstructed.append(outputs)
            lossess_noisy.append(criterion(outputs, imgs).tolist())

    return original_images, noisy_images, reconstructed, noisy_reconstructed, lossess_original, lossess_noisy
   

# Function to add noise to images
def noisify(img, noise_type, noise_level=0.3):
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


def plot_images(original, noisy, reconstructed, noisy_reconstructed, n=4):
    plt.figure(figsize=(10, 8))
    for i in range(n):
        # Original images
        plt.subplot(4, n, i+1)
        plt.imshow(original[i].numpy().reshape(28, 28), cmap='gray')
        plt.title("Original")
        plt.axis('off')

        # Noisy images
        plt.subplot(4, n, i+1+n)
        plt.imshow(noisy[i].numpy().reshape(28, 28), cmap='gray')
        plt.title("Noisy")
        plt.axis('off')

        # Reconstructed images
        plt.subplot(4, n, i+1+2*n)
        plt.imshow(reconstructed[i].numpy().reshape(28, 28), cmap='gray')
        plt.title("Reconstructed")
        plt.axis('off')

        # Reconstructed from noisy images
        plt.subplot(4, n, i+1+3*n)
        plt.imshow(noisy_reconstructed[i].numpy().reshape(28, 28), cmap='gray')
        plt.title("Reconstructed from Noisy")
        plt.axis('off')

    plt.tight_layout()
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


def plot_confusion_matrix(true_labels, predictions, class_names):
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
    train_autoencoder(autoencoder, train_loader, loss_fn, optimizer, test_loader, epochs=10)

    # evaluate performance
    test_images, noisy_test_images, reconstructed, noisy_reconstructed, losses_original, losses_noisy  = evaluate_autoencoder(autoencoder, test_loader, noisy_test_loader, loss_fn)
    # Plotting the images
    plot_images(test_images, noisy_test_images, reconstructed, noisy_reconstructed)
    # plot loss
    plot_loss(autoencoder)
    # plot treshold
    plot_tresh(losses_original, losses_noisy)


  

if __name__ == "__main__":
    main()