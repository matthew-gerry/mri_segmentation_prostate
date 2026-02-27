This is a self-study project to develop image segmentation models for identifying which pixels correspond to the prostate in MRI images.

Using the Promise12MSBench dataset available from the medsegbench library, I trained two models: one with a simple U-Net architecture trained from scratch, and another utilizing transfer learning, based on the deeplabv3_mobilenet_v3_large model available from torchvision.

For the latter, the base model needed to be modified so that the final output layer is a 1x1 convolution, such that the outputs correspond to logits that give a probability of being part of the prostate when passed through a sigmoid. I also found that unfreezing the last {\it four} layers of the ''classifier'' as well as the last two modules in the backbone, provided sufficient flexibility that model training was possible.

Here are some examples of images from the validation set, showing the ground truth contour of the prostate as well as the outline of each model's prediction. In the lower plots is a heat map of the probability the respective model assigned for each pixel to be part of the prostate.

![alt text](https://github.com/matthew-gerry/mri_segmentation_prostate/blob/main/figs/visualization_scratch.png?raw=true)
![alt text](https://github.com/matthew-gerry/mri_segmentation_prostate/blob/main/figs/visualization_tl.png?raw=true)


The resulting models reliably identified the location of the prostate and got its size approximately correct, but both struggled to get the shape quite right. It is worth noting that training was limited to a laptop CPU. Attempts to address this included incorporating a ''Hausdorff'' contribution to the loss function, which amplifies the significance of errors close to the boundary of the feature of interest. For both the model trained from scratch and the transfer learning-based model, it would also be interesting to investigate whether having more tunable parameters improves performance, either through a more elaborate neural network architecture, or through a transfer-based approach with more modules unfrozen.

Included is a pair of plots of the DICE score for the validation data through each epoch of training for both the from-scratch U-Net and the transfer learning-based model, showing convergence to a value around 0.94 for the former and 0.74 for the latter.
![alt text](https://github.com/matthew-gerry/mri_segmentation_prostate/blob/main/figs/val_DICE_vs_epoch.png?raw=true)
The high DICE score for the simple U-Net does seems somewhat at odds with visual inaccuracies we can see in the predicted contours of the prostate for a few example images from the validation set.