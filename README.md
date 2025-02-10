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

The training dataset for this project is stored in the `dataset` directory of this repository. This folder contains a sample representing the type of data used for model training. Specifically, the dataset comprises 2,000 images, each named in the format `data_xxxx.tiff`, where `xxxx` is a zero-padded number ranging from 0000 (or 0001) up to 2000, corresponding to each data point.

Each grayscale image, with pixel intensity values between 0 and 255, captures the fluorescence pattern produced by the atoms after excitation by lasers. Accompanying each image is a binary mask that serves as the ground truth for segmentation; for example, the file `data_0001.tiff` is paired with `truth_0001.tiff`.

This dataset was acquired in a single experimental run under fixed conditions, ensuring a consistent and reliable set of training data.
