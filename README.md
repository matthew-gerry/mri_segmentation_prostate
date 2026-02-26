To do:
- Write description of the project
- Attach example MRI images with model prediction

- Create some plots
    - Training + validation loss curves vs. epoch
    - (*) Same thing but DICE coefficient instead of overall loss (I like this better; maybe do validation only)
    - (*) Histogram of probabilities for prostate and background pixels (e.g., for all pixels labelled '0' in the mask, show the distribution of probabilties assigned by the model--these should cluster around zero)
    - Precision-recall curve (figure out what this is)
    - (*) Dice vs threshold - vary the decision threshold from 0 to 1 and show how the DICE varies with this choice
        - Do this and then modify the project scripts to use the optimal threshold
    - (*) Perimeter error (to address the elephant in the room that my model gets the shape wrong)
    - (*) Bland-Altman plot - agreement between predicted and ground truth area
    - Confusion matrix over all pixels, show as heat map

- Publish
