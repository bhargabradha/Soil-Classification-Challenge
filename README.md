# Soil Classification Challenge

This repository contains the code for a deep learning solution to classify soil types from images using PyTorch. This project is structured to accommodate multiple challenges or solution iterations, as shown in the directory structure below.

## Project Overview

The primary goal of this project is to build an image classification model capable of distinguishing between different soil categories (specifically, 4 classes). We achieve this by leveraging the power of transfer learning with a pre-trained Convolutional Neural Network (CNN) to achieve high accuracy and efficient training.

## Directory Structure

To keep the project organized and extensible for future challenges, the repository is structured as follows:

* `challenge-1/`: This directory contains the initial solution code for the soil classification task, including the main Python script.
* `LICENSE`: This file specifies the licensing terms for this project (MIT License).
* `README.md`: This file provides an overview of the entire repository and detailed instructions for each challenge.

## Challenge 1: PyTorch ResNet18 Classifier

This sub-directory (`challenge-1/`) holds the core Python script for the soil classification.

### Architecture

The solution utilizes a **ResNet18** model, which is a popular and efficient Convolutional Neural Network architecture. This model is loaded with weights pre-trained on the vast ImageNet dataset. The final classification layer of ResNet18 has been adapted to output 4 distinct categories, corresponding to our 4 soil types. This transfer learning approach allows us to leverage the extensive features learned from ImageNet, speeding up training and improving performance on our specific task.

### Data

The dataset used for this competition typically includes:

* `train_labels.csv`: A CSV file containing the unique image IDs and their corresponding class labels (0, 1, 2, or 3 for different soil types).
* `train/`: A directory containing all the training images.
* `test/`: A directory containing the test images for which predictions need to be generated.
* `test_ids.csv`: A CSV file listing the image IDs for the test set.

*(Note: These data files are usually provided via a competition platform like Kaggle and are not included directly in this repository due to their size. You will need to download them separately.)*

### Setup and Usage

Follow these steps to set up the project and run the soil classification model:

1.  **Clone the repository:**
    Open your terminal or command prompt and run:
    ```bash
    git clone [https://github.com/YOUR-USERNAME/Soil-Classification-Challenge.git](https://github.com/YOUR-USERNAME/Soil-Classification-Challenge.git)
    cd Soil-Classification-Challenge
    ```
    *(**Remember to replace `YOUR-USERNAME` with your actual GitHub username** and `Soil-Classification-Challenge` with your actual repository name if it differs.)*

2.  **Navigate to the challenge folder:**
    All the core Python code for this challenge is located in the `challenge-1` directory.
    ```bash
    cd challenge-1
    ```

3.  **Install dependencies:**
    Ensure you have Python and `pip` installed. Then, install the necessary libraries:
    ```bash
    pip install torch torchvision pandas scikit-learn matplotlib seaborn Pillow
    ```

4.  **Obtain Data:**
    * Download the competition data (the `soil_competition-2025` folder, which contains `train_labels.csv`, `train`, `test`, and `test_ids.csv`) from the relevant Kaggle competition page.
    * Place this `soil_competition-2025` folder in a location accessible by the script. **Important:** The current Python script (`soil_classifier_pytorch.py`) assumes the data is located at `/kaggle/input/soil-classification-2/soil_competition-2025` (a common path in Kaggle notebooks). If you are running this locally, you will likely need to **adjust the `img_path` and `test_img_path` variables** within your `soil_classifier_pytorch.py` file to point to the correct local paths where you've saved your downloaded data.

5.  **Run the training script:**
    Once dependencies are installed and data paths are correct, execute the main Python script:
    ```bash
    python soil_classifier_pytorch.py
    ```
    This script will:
    * Train the deep learning model.
    * Save the trained model weights as `resnet18_soil.pth`.
    * Generate and display plots showing training loss, validation accuracy, and validation F1-score over epochs.
    * Create a `submission.csv` file with predictions on the test set, ready for submission to the competition.

### Results

*(After running the script, you can update this section with your model's performance metrics, for example:)*
* **Final Validation Accuracy:** XX.XX%
* **Final Validation F1-Score (Weighted):** X.XXXX
* **Kaggle Public Leaderboard Score:** X.XXXX (if applicable)

*(You could also add a screenshot of your generated plots here!)*

## Contact

Feel free to reach out if you have any questions or feedback!

* **Your Name:** Bhargab Dey
* **Email:** bhargabdeyh@gmail.com
* **GitHub Profile:** [Your GitHub Profile Link Here - e.g., https://github.com/bhargabradha]



## Challenge-2: Soil Anomaly Detection (Autoencoder)

This section addresses a different aspect of soil classification: **anomaly detection**. Instead of multi-class classification, this challenge focuses on identifying "normal" soil images versus "anomalous" (non-soil or unusual) images.

### Architecture: Convolutional Autoencoder

* **Model Type:** A custom-built **Convolutional Autoencoder** using TensorFlow/Keras.
* **Purpose:** The autoencoder is trained in an unsupervised manner on a dataset consisting *only* of "normal" soil images. It learns to compress these images into a lower-dimensional representation (latent space) and then reconstruct them.
* **Anomaly Detection Mechanism:**
    * When a "normal" soil image is fed to the trained autoencoder, it can reconstruct it with a very low **reconstruction error** (the difference between the input and its reconstructed output).
    * However, if an "anomalous" image (e.g., non-soil, corrupted, or significantly different from the training data) is fed, the autoencoder struggles to reconstruct it accurately, leading to a **high reconstruction error**.
    * A statistical **threshold** (based on the mean and standard deviation of reconstruction errors on the training data) is used to classify new images:
        * **Error < Threshold:** Classified as "Soil" (normal).
        * **Error >= Threshold:** Classified as "Not Soil" (anomaly).

### Files in `Challenge-2` Folder:

* `soil_autoencoder_classifier.py`: Contains the full code for building, training, and evaluating the autoencoder, including data loading, preprocessing, model definition, training loop, error calculation, thresholding, and generating the submission file.