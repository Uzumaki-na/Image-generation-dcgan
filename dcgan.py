import os
import zipfile
import shutil
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import random

# Constants
BATCH_SIZE = 128  # Reduced batch size for stability
IM_SHAPE = (64, 64, 3)
LEARNING_RATE = 2e-4
LATENT_DIM = 100
EPOCHS = 1000
DATASET_PATH = "dataset/img_align_celeba"
CHECKPOINT_DIR = "checkpoints"
GENERATED_DIR = "generated"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Enable cuDNN for potential speedups
torch.backends.cudnn.benchmark = True

# Data Preparation
def download_and_prepare_data():
    """Downloads, unzips, and prepares the CelebA dataset."""
    if not os.path.exists('dataset'):
        print("Downloading dataset...")
        os.system('kaggle datasets download -d jessicali9530/celeba-dataset -p dataset')

    if not os.path.exists('dataset/celeba-dataset.zip'):
      print('Dataset Zip not downloaded')
      return

    print('Extracting dataset')
    with zipfile.ZipFile('dataset/celeba-dataset.zip', 'r') as zip_ref:
        zip_ref.extractall('dataset')

    # Move the images to the correct directory
    if not os.path.exists(DATASET_PATH):
        os.makedirs(DATASET_PATH)

    source_dir = 'dataset/img_align_celeba/img_align_celeba'
    if os.path.exists(source_dir): #Check if the source directory exists before moving
        print('Moving files...')
        for file in os.listdir(source_dir):
            shutil.move(os.path.join(source_dir, file), DATASET_PATH)

    #Remove old folder after move
    if os.path.exists('dataset/img_align_celeba'):
        shutil.rmtree('dataset/img_align_celeba')

    print("Dataset ready.")


# Dataset Class
class CelebADataset(Dataset):
    """Custom dataset class for CelebA."""
    def __init__(self, dataset_path, transform=None):
        self.dataset_path = dataset_path
        self.transform = transform
        self.image_paths = self._get_image_paths()

    def _get_image_paths(self):
        image_paths = []
        for file in os.listdir(self.dataset_path):
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                image_paths.append(os.path.join(self.dataset_path, file))
        return image_paths

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image

def load_dataset():
    """Loads and preprocesses the CelebA dataset."""
    transform = transforms.Compose([
        transforms.RandomResizedCrop(IM_SHAPE[0], scale=(0.8, 1.0)), # Random crop
        transforms.RandomHorizontalFlip(p=0.5),  # Random horizontal flip
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1), # Random color jitter
        transforms.RandomRotation(degrees=5), # Random rotation
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    dataset = CelebADataset(DATASET_PATH, transform=transform)
    train_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers = 8, pin_memory = True, drop_last = True) # Drop the last batch if it is smaller than batch size for consistency
    return train_loader


# Model Definitions
class Generator(nn.Module):
    """PyTorch Generator Model."""
    def __init__(self, latent_dim):
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(latent_dim, 4 * 4 * 512),
            nn.Unflatten(1, (512, 4, 4)),
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1, bias = False), # Removed Bias from Conv Transpose layer
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1, bias = False), # Removed Bias from Conv Transpose layer
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1, bias = False), # Removed Bias from Conv Transpose layer
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose2d(64, 3, kernel_size=4, stride=2, padding=1, bias = False), # Removed Bias from Conv Transpose layer
            nn.Tanh()
        )

    def forward(self, x):
        return self.model(x)

class Discriminator(nn.Module):
    """PyTorch Discriminator Model."""
    def __init__(self):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1, bias = False), # Removed bias from the conv layer
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
             nn.Flatten(),
            nn.Linear(128* 16* 16 , 1), # Changed the conv layer to linear
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x)


# Training Function
def train_gan(generator, discriminator, train_loader, d_optimizer, g_optimizer, loss_fn, epochs, latent_dim, device):
    """Trains the GAN model."""
    os.makedirs(GENERATED_DIR, exist_ok=True)
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    print(f"Training on {device}...")

    fixed_noise = torch.randn(36, latent_dim, device = device)

    generator.train()
    discriminator.train()

    for epoch in range(epochs):
        loop = tqdm(train_loader, desc = f"Epoch {epoch+1}/{epochs}")
        d_loss_epoch = 0
        g_loss_epoch = 0
        for i, real_images in enumerate(loop):

            real_images = real_images.to(device)
            batch_size = real_images.size(0)

            # ---------------------
            # Train Discriminator
            # ---------------------
            d_optimizer.zero_grad()

            # Real images loss
            real_labels = torch.ones((batch_size, ), device=device)
            # Add noise to real labels
            real_labels += 0.25 * torch.rand(real_labels.shape, device=device) * 2 - 1

            real_predictions = discriminator(real_images).squeeze(1) #squeeze here now to make sure the dimensions match
            d_loss_real = loss_fn(real_predictions, real_labels)

            # Fake images loss
            random_noise = torch.randn(batch_size, latent_dim, device=device)
            fake_images = generator(random_noise)

            fake_labels = torch.zeros((batch_size, ), device=device)
            # Add noise to fake labels
            fake_labels += 0.25 * torch.rand(fake_labels.shape, device=device)

            fake_predictions = discriminator(fake_images.detach()).squeeze(1) #squeeze here now to make sure the dimensions match
            d_loss_fake = loss_fn(fake_predictions, fake_labels)


            d_loss = d_loss_real + d_loss_fake
            d_loss.backward()
            d_optimizer.step()
            d_loss_epoch += d_loss.item()

            # ---------------------
            # Train Generator
            # ---------------------
            g_optimizer.zero_grad()

            random_noise = torch.randn(batch_size, latent_dim, device=device)
            fake_images = generator(random_noise)
            fake_predictions = discriminator(fake_images).squeeze(1) #squeeze here now to make sure the dimensions match

            flipped_fake_labels = torch.ones((batch_size, ), device=device)
            g_loss = loss_fn(fake_predictions, flipped_fake_labels)

            g_loss.backward()
            g_optimizer.step()
            g_loss_epoch += g_loss.item()


            loop.set_postfix(d_loss = d_loss.item() , g_loss = g_loss.item())
        # Save generated images
        generator.eval()
        with torch.no_grad():
            gen_images = generator(fixed_noise).cpu().detach()
            gen_images = (gen_images * 0.5 + 0.5).clamp(0, 1)
            fig, axs = plt.subplots(6, 6, figsize=(16, 16))
            for k, ax in enumerate(axs.flat):
                ax.imshow(gen_images[k].permute(1, 2, 0))
                ax.axis('off')
            plt.savefig(os.path.join(GENERATED_DIR, f"gen_images_epoch_{epoch + 1}.png"))
            plt.close()
        generator.train()

        print(f"Epoch {epoch + 1}/{epochs}, Discriminator Loss: {d_loss_epoch/len(train_loader):.4f}, Generator Loss: {g_loss_epoch/len(train_loader):.4f}")

        # Save model checkpoints
        torch.save(generator.state_dict(), os.path.join(CHECKPOINT_DIR, f'generator_epoch_{epoch+1}.pth'))
        torch.save(discriminator.state_dict(), os.path.join(CHECKPOINT_DIR, f'discriminator_epoch_{epoch+1}.pth'))


# Main
if __name__ == "__main__":
    # Download and prepare data if the directory doesn't exists.
    if not os.path.exists(DATASET_PATH):
        download_and_prepare_data()

    train_loader = load_dataset()

    generator = Generator(LATENT_DIM).to(DEVICE)
    discriminator = Discriminator().to(DEVICE)

    g_optimizer = optim.Adam(generator.parameters(), lr=LEARNING_RATE, betas=(0.5, 0.999))
    d_optimizer = optim.Adam(discriminator.parameters(), lr=LEARNING_RATE, betas=(0.5, 0.999))

    loss_fn = nn.BCELoss()

    train_gan(generator, discriminator, train_loader, d_optimizer, g_optimizer, loss_fn, EPOCHS, LATENT_DIM, DEVICE)