This is a self-study project to develop image segmentation models, implemented as a cli tool.

Install the latest version of the cli tool (with the train command complete, other commands still to come) by running:

```pip install "image-seg[promise12] @ git+https://github.com/matthew-gerry/mri_segmentation_prostate.git@main"```

The train command can then be run, along with a config.yaml file formatted according to the example provided, using:
```image-seg train --config <path-to-config-yaml-file>```

The model parameters will then be saved to the specified directory. It can be referenced for model evaulation using the same config.yaml file, via:
``` image-seg evaluate --config <path-to-config-yaml-file> ``` 

Additionally, a few visualizations, namely, the original images overlaid with contours of the predicted region, as well as the history of the DICE coefficient through training, and a Bland-Altman plot, are available through the visualize command:
``` image-seg visualize --config <path-to-config-yaml-file> ``` 


Using the Promise12MSBench dataset available from the medsegbench library (https://medsegbench.github.io/), I developed scripts to train two models to identify pixels making up the prostate in MRI images: one with a simple U-Net architecture trained from scratch, and another utilizing transfer learning, based on the deeplabv3_mobilenet_v3_large model available from torchvision. A future version of this tool may include the capacity to use locally saved image data sources rather than being limited to image datasets available through medsegbench.

For the latter, the base model needed to be modified so that the final output layer is a 1x1 convolution, such that the outputs correspond to logits that give a probability of being part of the prostate when passed through a sigmoid. I also found that unfreezing the last four layers of the ''classifier'' as well as the last two modules in the backbone, provided sufficient flexibility that training could result in a model with some effectiveness on the dataset.

Here are some examples of images from the validation set, showing the ground truth contour of the prostate as well as the outline of each model's prediction. In the lower panels is a heat map of the probability the respective model assigned for each pixel to be part of the prostate, with a probability of 0.5 being used as the cutoff amounting to the boundaries shown in the images.

![visualization of model predictions and probability heatmaps for the U-Net trained from scratch](https://github.com/matthew-gerry/mri_segmentation_prostate/blob/main/figs/predictions_unet.png?raw=true)
![visualization of model predictions and probability heatmaps for the transfer learning-based model](https://github.com/matthew-gerry/mri_segmentation_prostate/blob/main/figs/predictions_mnv3.png?raw=true)


The resulting models reliably identified the location of the prostate and got its size approximately correct, but both struggled to get the shape quite right. Attempts to address this included incorporating a ''Hausdorff'' contribution to the loss function, which amplifies the significance of errors close to the boundary of the feature of interest. It is worth noting that training was limited to a laptop CPU. For both the model trained from scratch and the transfer learning-based model, it would be interesting to investigate whether having more tunable parameters improves performance, either through a more elaborate neural network architecture, or through a transfer-based approach with more modules unfrozen. GPU-acceleration would make it more feasible to pursue these directions.

Included is a plot of the DICE score for the validation data through each epoch of training for the from-scratch U-Net model, showing convergence to a value around 0.85.

![Plots of the DICE coefficient on the validation data vs training epoch for each model](https://github.com/matthew-gerry/mri_segmentation_prostate/blob/main/figs/val_DICE_vs_epoch_unet.png?raw=true)

Also shown is a Bland-Altman plot showing how the areas of the prostate as predicted by the U-Net model (from scratch) compare to the ground truth for the validation data. The plot indicates that the model tends to underestimate the size of the prostate in the images, with a few outliers contibuting especially to bringing the overall bias down. These outliers represent cases in which few or no pixels were included in the prediction, despite the fact that a DICE coefficient-based loss was used to try and account for the imbalance between non-prostate and prostate pixels in the ground truth dataset.

![Bland-Altman plot showing the performance of the U-Net trained from scratch](https://github.com/matthew-gerry/mri_segmentation_prostate/blob/main/figs/bland_altman_unet.png?raw=true)
