'''
Main script for the prostate MRI image segmentation tool. Calls data loading, and model training functions, and saves the trained model.

Matthew Gerry
February 2026
'''

import torch
from torch.utils.data import DataLoader
import time

from prostatemri_dataset import load_images, MRIDataset
from mriseg_model import SimpleUNet
from losses import combined_loss

def main():
    # Load and prepare data
    train_images = load_images("train")

    print(f"Loaded {len(train_images)} training images.")

    # Convert to PyTorch datasets
    train_dataset = MRIDataset(train_images)

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)

    # Initialize model and optimizer 
    model = SimpleUNet()
    # Check the properties of the model such as the number of tunable parameters
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total tunable parameters in the model: {total_params}")

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # Train the model
    num_epochs = 15
    start_time = time.time()
    for epoch in range(num_epochs):
        model.train() # This line sets the model to training mode, which is important for layers like dropout and batch normalization to behave correctly during training
        running_loss = 0.0
        
        for images, masks in train_loader:
            optimizer.zero_grad()
            outputs = model(images)

            # Update weight of boundary loss term based on a schedule
            bdy_weight = min(1.0, epoch/5) # Works itself up to 1 over first 5 epochs, since earlier on it is better to focus on getting the overall shape right before heavily penalizing boundary errors
    
            loss = combined_loss(outputs, masks, boundary_weight=bdy_weight)

            loss.backward() # Differentiate loss function
            optimizer.step() # Update model parameters
            running_loss += loss.item() * images.size(0) # Multiply by batch size to get total loss for the batch

        avg_train_loss = running_loss / len(train_loader.dataset)
        print(f"Epoch {epoch+1}/{num_epochs}, Training Loss: {avg_train_loss:.4f}")

        training_time = time.time() - start_time
        print(f"Total training time so far: {training_time//60:.0f}m {training_time%60:.0f}s")
    
    # Save the trained model
    torch.save(model.state_dict(), "prostate_unet.pth")


if __name__ == "__main__":
    main()