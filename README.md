# Lung Segmentation using U-Net

This repo contains a complete pipeline for training a Deep Learning model to perform lung segmentation on Chest X-Ray (CXR) images. It utilizes a **U-Net architecture** built with TensorFlow and Keras to accurately map and extract lung boundaries from medical imagery. 

## Dataset
This project uses the **Montgomery County X-ray Set**. 
* **Dataset Link:** [Link](https://academictorrents.com/details/ac786f74878a5775c81d490b23842fd4736bfe33)

The dataset provides raw Chest X-rays alongside manually annotated masks for both the left and right lungs. The script automatically merges these individual masks into a unified ground-truth mask for training.

## Features
* **Custom U-Net Architecture:** Features a 4-tier encoder-decoder network with skip connections for precise spatial localization.
* **Automated Data Pipeline:** Loads, resizes (to 256x256), normalizes, and combines left/right lung masks automatically.
* **Data Caching:** Saves preprocessed training and validation sets as `.npy` arrays to significantly speed up future training runs.
* **Robust Training Callbacks:** Implements `EarlyStopping`, `ReduceLROnPlateau`, and `ModelCheckpoint` to prevent overfitting and capture the best model weights.
* **Visual Verification:** Includes an OpenCV-based inference function to predict and display the segmentation mask over a test image.

## Requirements

Ensure you have Python 3.7+ installed. You will need the following libraries:

```bash
pip install tensorflow opencv-python numpy scikit-learn tqdm
