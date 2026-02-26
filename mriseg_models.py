'''
U-net definition for use in prostate segmentation task.
'''

import numpy as np
import torch
import torch.nn as nn
from torchvision.models.segmentation import deeplabv3_mobilenet_v3_large

# Let's set up a simple UNet model for segmentation
class SimpleUNet(nn.Module):
    '''
    Simple U-Net for image segmentation, built from scratch. Consists of an encoder-decoder architecture with skip connections, using convolutional blocks with batch normalization and ReLU activations.
    '''
    def __init__(self, ):
        super().__init__()

        # Apply a few encoding layers with pooling
        self.enc1 = self._conv_block(1, 32)
        self.pool1 = nn.MaxPool2d(2)
        self.enc2 = self._conv_block(32, 64)
        self.pool2 = nn.MaxPool2d(2)
        self.bottleneck = self._conv_block(64, 128)
        
        # Decode back to a single channel using convolutions as well
        self.up2 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.dec2 = self._conv_block(128 + 64, 64)
        self.up1 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.dec1 = self._conv_block(64 + 32, 32)

        self.final = nn.Conv2d(32, 1, kernel_size=1)

    # Define the convoltuional layers, including batch normalization and ReLU activation at each step
    def _conv_block(self, in_ch, out_ch):
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            )

    # Apply each block defined above in sequence
    def forward(self, x):
        e1 = self.enc1(x)
        p1 = self.pool1(e1)
        e2 = self.enc2(p1)
        p2 = self.pool2(e2)
        b = self.bottleneck(p2)

        d2 = self.up2(b)
        d2 = torch.cat([d2, e2], dim=1)
        d2 = self.dec2(d2)

        d1 = self.up1(d2)
        d1 = torch.cat([d1, e1], dim=1)
        d1 = self.dec1(d1)

        return self.final(d1)


class TLDeepLabV3MobileNet(nn.Module):
    '''
    Transfer-learning segmentation model (DeepLabV3 + MobileNetV3-Large) adapted for grayscale inputs, with configurable unfreezing of backbone and classifier layers for fine-tuning to the dataset.

    Steps performed:
      1) Load pretrained model.
      2) Replace first 3->1 Conv2d in the backbone (initialize weights as RGB-mean).
      3) Replace classifier's final conv to output `num_classes` (default 1 for binary).
      4) Freeze all parameters.
      5) Unfreeze selected backbone stages (by substring match, e.g., '5'/'6').
      6) Unfreeze last K modules in classifier head.

    Forward returns torchvision-style dict: {'out': logits, ...}
    '''

    def __init__(
        self,
        backbone_unfreeze_substrings = ("5", "6"),
        classifier_unfreeze_last_K = 4
    ):
        super().__init__()

        # Load pretrained
        self.model = deeplabv3_mobilenet_v3_large(weights='DEFAULT')

        # Replace first Conv2d (3->1) in the backbone, robustly
        target_name, first_conv = None, None
        for name, module in self.model.backbone.named_modules():
            if isinstance(module, nn.Conv2d) and module.in_channels == 3:
                target_name, first_conv = name, module
                break
        # if first_conv is None:
        #     raise RuntimeError("Could not find a 3-channel Conv2d in the backbone.")

        new_conv = nn.Conv2d(
            in_channels=1,
            out_channels=first_conv.out_channels,
            kernel_size=first_conv.kernel_size,
            stride=first_conv.stride,
            padding=first_conv.padding,
            dilation=first_conv.dilation,
            groups=1,  # keep standard conv for the first layer
            bias=(first_conv.bias is not None),
            padding_mode=getattr(first_conv, "padding_mode", "zeros"),
        )
        with torch.no_grad():
            new_conv.weight.copy_(first_conv.weight.mean(dim=1, keepdim=True))
            if first_conv.bias is not None and new_conv.bias is not None:
                new_conv.bias.copy_(first_conv.bias)

        # Set it back into the backbone at the same module path
        parent = self.model.backbone
        parts = target_name.split(".")
        for p in parts[:-1]:
            parent = getattr(parent, p)
        setattr(parent, parts[-1], new_conv)

        # Replace classifier last layer to a single output channel (binary classification)
        in_ch = self.model.classifier[-1].in_channels
        self.model.classifier[-1] = nn.Conv2d(in_ch, 1, kernel_size=1)

        # Freeze everything
        for p in self.model.parameters():
            p.requires_grad = False

        # Unfreeze selected backbone stages by name substring (e.g., '5'/'6')
        self._unfreeze_backbone_by_name_substrings(backbone_unfreeze_substrings)

        # Unfreeze last K modules of classifier head
        self._unfreeze_classifier_tail(k_last=classifier_unfreeze_last_K)

    def _unfreeze_backbone_by_name_substrings(self, substrings):
        ''' UNFREEZE BACKBONE PARAMETERS IN BLOCKS CORRESPONDING TO THE NUMERICAL LABELS GIVEN (E.G. '5', '6') '''
        if not substrings:
            return
        for name, p in self.model.named_parameters():
            if "backbone" in name and any(s in name for s in substrings):
                p.requires_grad = True

    def _unfreeze_classifier_tail(self, k_last):
        '''UNFREEZE THE LAST K MODULES OF THE CLASSIFIER HEAD, ALWAYS ENSURE FINAL 1X1 CONVOLUTION IS TRAINABLE'''
        # Always ensure the final 1x1 is trainable
        for p in self.model.classifier[-1].parameters():
            p.requires_grad = True

        if k_last <= 0:
            return

        # Unfreeze last k modules of classifier (e.g., projection + last convs)
        modules = list(self.model.classifier.children())
        for m in modules[-k_last:]:
            for p in m.parameters():
                p.requires_grad = True

    def forward(self, x):
        # Keeps torchvision behavior: returns dict with key 'out'
        return self.model(x)

    def trainable_parameters(self):
        '''CONVENIENCE: ONLY PARAMETERS WITH requires_grad=True.'''
        return (p for p in self.parameters() if p.requires_grad)
