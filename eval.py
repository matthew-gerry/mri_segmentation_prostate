'''
Evaluate the model for the prostate segmentation task, using the validation set and computing metrics such as Dice coefficient and boundary-specific metrics.
'''

import torch
from torch.utils.data import DataLoader
from prostatemri_dataset import load_images, MRIDataset
from mriseg_model import SimpleUNet
from losses import combined_loss
import utils

def main(model, dataloader):
    model.eval() # Set model to evaluation mode (important for layers like dropout and batch normalization)
   
    # Load in the validation data
    val_images = load_images("val")
    val_dataset = MRIDataset(val_images)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)

    # Initialize metrics
    total_loss = 0.0
    total_accuracy = 0.0
    total_dice = 0.0
    num_samples = 0

    with torch.no_grad(): # No need to compute gradients during evaluation
        for images, masks in val_loader:
            outputs = model(images)

            # Compute combined loss for monitoring (not used for optimization here)
            batch_loss = combined_loss(outputs, masks, boundary_weight=1.0)
            total_loss += batch_loss.item() * images.size(0)

            # Compute accuracy for this batch
            batch_accuracy = utils.accuracy(outputs, masks)
            total_accuracy += batch_accuracy.item() * images.size(0)

            # Compute Dice coefficient for this batch
            batch_dice = utils.dice_coefficient(outputs, masks)
            total_dice += batch_dice.item() * images.size(0)

            num_samples += images.size(0)

    avg_loss = total_loss / num_samples
    avg_accuracy = total_accuracy / num_samples
    avg_dice = total_dice / num_samples
    print(f"Average Loss on Validation Set: {avg_loss:.4f}")
    print(f"Average Accuracy on Validation Set: {avg_accuracy*100:.2f}%")
    print(f"Average Dice Coefficient on Validation Set: {avg_dice:.4f}")

if __name__ == "__main__":
    # Load the trained model
    model = SimpleUNet()
    model.load_state_dict(torch.load("prostate_unet.pth"))

    # Run evaluation
    main(model, None)