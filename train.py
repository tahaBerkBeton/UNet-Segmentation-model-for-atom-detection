import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from PIL import Image
import numpy as np
from glob import glob
from tqdm import tqdm
import json
import matplotlib.pyplot as plt

# Determine the project root directory (the directory of this file)
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

# Model Architecture Definition
class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ConvBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        return x

class GridNetU(nn.Module):
    def __init__(self):
        super(GridNetU, self).__init__()
        
        # Encoder blocks
        self.enc1 = ConvBlock(1, 32)
        self.enc2 = ConvBlock(32, 64)
        self.enc3 = ConvBlock(64, 128)
        self.enc4 = ConvBlock(128, 256)
        
        # Bridge
        self.bridge = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5)
        )
        
        # Decoder blocks
        self.dec4 = ConvBlock(512 + 256, 256)
        self.dec3 = ConvBlock(256 + 128, 128)
        self.dec2 = ConvBlock(128 + 64, 64)
        self.dec1 = ConvBlock(64 + 32, 32)
        
        # Max pooling and upsampling
        self.pool = nn.MaxPool2d(2)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        
        # Final output
        self.final = nn.Conv2d(32, 1, kernel_size=1)
        
    def forward(self, x):
        # Encoder
        enc1 = self.enc1(x)
        enc2 = self.enc2(self.pool(enc1))
        enc3 = self.enc3(self.pool(enc2))
        enc4 = self.enc4(self.pool(enc3))
        
        # Bridge
        bridge = self.bridge(self.pool(enc4))
        
        # Decoder with skip connections
        dec4 = self.dec4(torch.cat([self.upsample(bridge), enc4], dim=1))
        dec3 = self.dec3(torch.cat([self.upsample(dec4), enc3], dim=1))
        dec2 = self.dec2(torch.cat([self.upsample(dec3), enc2], dim=1))
        dec1 = self.dec1(torch.cat([self.upsample(dec2), enc1], dim=1))
        
        # Final output (no sigmoid - it's included in BCEWithLogitsLoss)
        out = self.final(dec1)
        return out

class AtomDataset(Dataset):
    def __init__(self, data_dir):
        self.data_files = sorted(glob(os.path.join(data_dir, "data_*.tiff")))
        self.truth_files = sorted(glob(os.path.join(data_dir, "truth_*.tiff")))
        
    def __len__(self):
        return len(self.data_files)
    
    def __getitem__(self, idx):
        # Load data and truth images
        data = np.array(Image.open(self.data_files[idx])).astype(np.float32)
        truth = np.array(Image.open(self.truth_files[idx])).astype(np.float32)
        
        # Normalize data to [0, 1]
        data = data / 265.0
        
        # Convert to torch tensors and add channel dimension
        data = torch.from_numpy(data).unsqueeze(0)
        truth = torch.from_numpy(truth).unsqueeze(0)
        
        return data, truth

def train_model(model, train_loader, val_loader, device, num_epochs=30):
    # Calculate class weights for loss function
    pos_weight = torch.tensor([97/3])  # Approximately 2.89% positive class
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight.to(device))
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    
    best_val_loss = float('inf')
    train_losses = []
    val_losses = []
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        train_steps = 0
        
        for inputs, targets in tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}'):
            inputs = inputs.to(device)
            targets = targets.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            train_steps += 1
        
        avg_train_loss = train_loss / train_steps
        train_losses.append(avg_train_loss)
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_steps = 0
        
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs = inputs.to(device)
                targets = targets.to(device)
                
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                
                val_loss += loss.item()
                val_steps += 1
        
        avg_val_loss = val_loss / val_steps
        val_losses.append(avg_val_loss)
        
        # Print metrics
        print(f'Epoch {epoch+1}/{num_epochs}:')
        print(f'Training Loss: {avg_train_loss:.4f}')
        print(f'Validation Loss: {avg_val_loss:.4f}')
        
        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': best_val_loss,
            }, os.path.join(ROOT_DIR, 'best_model.pth'))

    # Save loss plot
    epochs = range(1, num_epochs + 1)
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, train_losses, label='Training Loss')
    plt.plot(epochs, val_losses, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss Over Epochs')
    plt.legend()
    plt.grid()
    plt.savefig(os.path.join(ROOT_DIR, 'training_validation_loss.png'))
    print("Training and validation loss plot saved as 'training_validation_loss.png'")

def main():
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load dataset (using relative path)
    dataset_dir = os.path.join(ROOT_DIR, "dataset")
    dataset = AtomDataset(dataset_dir)
    
    # Split dataset
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    # Save validation filenames to JSON (using relative path)
    val_filenames = [dataset.data_files[idx] for idx in val_dataset.indices]
    json_path = os.path.join(ROOT_DIR, "validation_filenames.json")
    with open(json_path, 'w') as f:
        json.dump(val_filenames, f, indent=4)
    print(f"Validation filenames saved to {json_path}")
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=24, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False, num_workers=4)
    
    # Initialize model
    model = GridNetU().to(device)
    
    # Train model
    train_model(model, train_loader, val_loader, device)

if __name__ == "__main__":
    main()
