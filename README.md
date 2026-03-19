This is a self-study project to develop image segmentation models. The program is currently used by running the Python scripts, with the configuration set and hyperparameter adjustments made by editing the scripts themselves. In the feature branch I am working on turning it into a cli tool. Currently, the train command of the cli tool is complete, but additional commands are still being developed (evaluate, predict, visualizations, etc.).

Using the Promise12MSBench dataset available from the medsegbench library (https://medsegbench.github.io/), I trained two models to identify pixels making up the prostate in MRI images: one with a simple U-Net architecture trained from scratch, and another utilizing transfer learning, based on the deeplabv3_mobilenet_v3_large model available from torchvision.

For the latter, the base model needed to be modified so that the final output layer is a 1x1 convolution, such that the outputs correspond to logits that give a probability of being part of the prostate when passed through a sigmoid. I also found that unfreezing the last four layers of the ''classifier'' as well as the last two modules in the backbone, provided sufficient flexibility that training could result in a model with some effectiveness on the dataset.

Here are some examples of images from the validation set, showing the ground truth contour of the prostate as well as the outline of each model's prediction. In the lower panels is a heat map of the probability the respective model assigned for each pixel to be part of the prostate, with a probability of 0.5 being used as the cutoff amounting to the boundaries shown in the images.

![visualization of model predictions and probability heatmaps for the U-Net trained from scratch](https://github.com/matthew-gerry/mri_segmentation_prostate/blob/main/figs/visualization_scratch.png?raw=true)
![visualization of model predictions and probability heatmaps for the transfer learning-based model](https://github.com/matthew-gerry/mri_segmentation_prostate/blob/main/figs/visualization_tl.png?raw=true)


The resulting models reliably identified the location of the prostate and got its size approximately correct, but both struggled to get the shape quite right. Attempts to address this included incorporating a ''Hausdorff'' contribution to the loss function, which amplifies the significance of errors close to the boundary of the feature of interest. It is worth noting that training was limited to a laptop CPU. For both the model trained from scratch and the transfer learning-based model, it would be interesting to investigate whether having more tunable parameters improves performance, either through a more elaborate neural network architecture, or through a transfer-based approach with more modules unfrozen. GPU-acceleration would make it more feasible to pursue these directions.

Included is a pair of plots of the DICE score for the validation data through each epoch of training for both the from-scratch U-Net and the transfer learning-based model, showing convergence to a value around 0.94 for the former and 0.74 for the latter.

![Plots of the DICE coefficient on the validation data vs training epoch for each model](https://github.com/matthew-gerry/mri_segmentation_prostate/blob/main/figs/val_DICE_vs_epoch.png?raw=true)

The high DICE score for the simple U-Net does seem somewhat at odds with visual inaccuracies we can see in the predicted contours of the prostate for a few example images from the validation set.

Also shown is a Bland-Altman plot showing how the areas of the prostate as predicted by the U-Net model (from scratch) compare to the ground truth. The plot indicates that the model tends to underestimate the size of the prostate in the images, with a few outliers contibuting especially to bringing the overall bias down. These outliers represent cases in which few or no pixels were included in the prediction, despite the fact that a DICE coefficient-based loss was used to try and account for the imbalance between non-prostate and prostate pixels in the ground truth dataset.

![Bland-Altman plot showing the performance of the U-Net trained from scratch](https://github.com/matthew-gerry/mri_segmentation_prostate/blob/main/figs/val_bland_altman_scratch.png?raw=true)

Install the latest version of the cli tool (with the train command complete, other commands still to come) by running:

```pip install "image-seg[promise12] @ git+https://github.com/matthew-gerry/mri_segmentation_prostate.git@feature"```

The train command can then be run, along with a config.yaml file formatted according to the example provided, using:
```image-seg train --config <path-to-config-yaml-file>```