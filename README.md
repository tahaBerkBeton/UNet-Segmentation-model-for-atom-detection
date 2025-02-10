# UNet Segmentation Model for Atom Detection

## Introduction  

In this project, we address the problem of detecting atomic positions from fluorescence images captured in an optical lattice. The atoms are excited by lasers in multiple directions, and this excitation, along with the resulting scattering effects, produces fluorescence. The emitted light forms a distinct pattern on the detector, allowing us to observe the spatial distribution of atoms.  

Our goal is to segment these fluorescence images to accurately identify the exact positions of the atoms responsible for the observed patterns. To achieve this, we frame the task as an image segmentation problem and employ a U-Net-based architecture, inspired by the well-established U-Net model introduced by Ronneberger et al. This model's encoder-decoder structure with skip connections makes it particularly well-suited for precise localization in our application.  
