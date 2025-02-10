# UNet Segmentation Model for Atom Detection

## Introduction

In this project, we address the problem of detecting atomic positions from fluorescence images captured in an optical lattice. The atoms are excited by lasers in multiple directions, and this excitation—along with the resulting scattering effects—produces fluorescence. The emitted light forms a distinct pattern on the detector, allowing us to observe the spatial distribution of atoms.

Our goal is to segment these fluorescence images to accurately identify the exact positions of the atoms responsible for the observed patterns. To achieve this, we frame the task as an image segmentation problem and employ a U-Net-based architecture, inspired by the well-established U-Net model introduced by Ronneberger et al. This model's encoder-decoder structure with skip connections makes it particularly well-suited for precise localization in our application.

## Architecture Chosen

The architecture chosen for this project is a U-Net based model, specifically a variant referred to as `GridNetU`. This design is tailored for segmentation tasks and offers several key benefits for atom detection:

- **Encoder**:  
  The model begins with an encoder composed of four convolutional blocks (`ConvBlock`). These blocks extract features at different scales by progressively increasing the number of channels (from 1 to 32, 32 to 64, 64 to 128, and 128 to 256) while reducing the spatial resolution via max pooling.

- **Bridge**:  
  Following the encoder, a bridge module further processes the features using additional convolutional layers, batch normalization, ReLU activations, and dropout. This stage acts as a bottleneck, capturing high-level abstractions from the input data.

- **Decoder**:  
  The decoder mirrors the encoder with four corresponding blocks that perform upsampling and combine features from the encoder through skip connections. This fusion of low-level and high-level information enables the network to reconstruct detailed segmentation maps, crucial for accurately pinpointing the positions of atoms.

- **Final Output**:  
  A 1x1 convolution layer is applied at the end of the decoder to produce the segmentation map. (Note: The sigmoid activation is handled within the loss function, using BCEWithLogitsLoss.)

For a detailed view of the architecture implementation, please refer to the `train.py` file in this repository.

## Dataset

The training dataset for this project is stored in the `dataset` directory of this repository. This folder contains a sample representing the type of data used for model training. Specifically, the dataset comprises 2,000 images, each named in the format `data_xxxx.tiff`, where `xxxx` is a zero-padded number ranging from 0001 up to 2000, corresponding to each data point.

Each grayscale image, with pixel intensity values between 0 and 255, captures the fluorescence pattern produced by the atoms after excitation by lasers. Accompanying each image is a binary mask that serves as the ground truth for segmentation; for example, the file `data_0001.tiff` is paired with `truth_0001.tiff`.

This dataset was acquired in a single experimental run under fixed conditions.

## Training

The model is trained using a standard deep learning pipeline implemented in PyTorch. We perform an 80/20 split of the dataset into training and validation sets and use data loaders to efficiently feed batches of images and their corresponding masks into the model.

To address the class imbalance in the segmentation masks, we employ the `BCEWithLogitsLoss` loss function with a custom positive class weight (`pos_weight=97/3`). This helps the model to better learn from the less frequent positive pixels. The optimizer used is Adam, with a learning rate of 1e-4, which effectively updates the network parameters during training.

Training is conducted over 30 epochs. During each epoch, the model undergoes forward propagation, loss computation, backpropagation, and parameter updates. The validation set is used to monitor the model's performance after each epoch, and the best model (with the lowest validation loss) is saved. Additionally, a plot of the training and validation loss over epochs is generated to visualize the learning progress.

For a more detailed view of the training procedure, please refer to the `train.py` file.

## Test and Results

The performance of the trained model is evaluated using two scripts: `test.py` and `statistics_val_dataset.py`.

- **Single Image Evaluation (`test.py`)**:  
  This script randomly selects a validation image and its corresponding ground truth mask. The image is preprocessed and fed into the model, which outputs a raw prediction. A threshold (0.5) is applied to convert the raw output into a binary mask. Key performance metrics—such as precision, recall, F1 score, accuracy, and specificity—are computed. Additionally, the script generates a visualization that displays the input data, ground truth, raw prediction, and binary prediction with error regions highlighted. Users can refer to the `test.py` file for a detailed look at the evaluation process on individual images.

- **Overall Validation Evaluation (`statistics_val_dataset.py`)**:  
  To obtain a comprehensive assessment of the model's performance, this script processes the entire validation dataset. It aggregates the confusion matrix components (true positives, false positives, true negatives, and false negatives) across all validation samples and computes the overall performance metrics. For instance, the evaluation on our validation set produced the following metrics:
  
  ```json
  {
      "metrics": {
          "precision": 0.9931077647512676,
          "recall": 0.9984346316261389,
          "f1": 0.995764074185028,
          "accuracy": 0.9997544725406805,
          "specificity": 0.999793756747181
      }
  }

The above overall metrics demonstrate the high performance of our model. With a precision of 0.9931, recall of 0.9984, and an F1 score of 0.9958, the model reliably identifies atom positions with minimal false positives and negatives. The exceptional accuracy (99.9754%) and specificity (99.9794%) further confirm the robustness and reliability of our approach under fixed experimental conditions.
- **ROC Curve (`roc.py`)**:  
Additionally, the `roc.py` script provides further insights by computing the ROC curve across a range of detection thresholds. By evaluating the true positive rate (TPR) and false positive rate (FPR) at 10 evenly spaced thresholds between 0 and 1, the ROC curve illustrates that the model maintains high sensitivity while keeping false detections to a minimum. This analysis reinforces the quality of our results and validates the effectiveness of our U-Net-based architecture for the atom detection task.

