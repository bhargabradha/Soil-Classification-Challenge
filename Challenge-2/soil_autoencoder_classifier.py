import os
import numpy as np
import pandas as pd
from PIL import Image, UnidentifiedImageError # PIL for image loading, UnidentifiedImageError for handling corrupted/unreadable images
from sklearn.model_selection import train_test_split # For splitting data into training and validation sets
from tensorflow.keras.models import Model # To build the Keras model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, Flatten, Dense, Reshape # Keras layers for building the autoencoder
from tensorflow.keras.preprocessing.image import ImageDataGenerator # For real-time data augmentation
from tensorflow.keras import backend as K # Keras backend utilities (less used in this script but good practice)
import matplotlib.pyplot as plt # For plotting training history and error distributions
import sys # For system-specific parameters and functions (e.g., sys.exit)
from scipy.stats import norm # For statistical thresholding (fitting a normal distribution to errors)

# --- 0. Dynamic Path Discovery and Configuration ---
# Reasoning: This section ensures the script can locate the dataset files
# dynamically, which is crucial for reproducibility, especially in environments
# like Kaggle where dataset paths might vary. It makes the code portable.
print("--- 0. Discovering Dataset Paths ---")
BASE_PATH = None
TRAIN_DIR = None
TEST_DIR = None
TRAIN_LABELS_CSV = None
TEST_IDS_CSV = None
SUBMISSION_CSV_TEMPLATE = None

potential_base_paths = []
# Reasoning: Iterates through common Kaggle input directories to find the dataset.
for dirname, subdirs, filenames in os.walk('/kaggle/input'):
    if 'train_labels.csv' in filenames and 'train' in subdirs:
        potential_base_paths.append(dirname)

if not potential_base_paths:
    # Reasoning: Critical error handling if dataset isn't found, preventing further execution.
    print("CRITICAL ERROR: No dataset containing 'train_labels.csv' and a 'train' directory found in /kaggle/input.")
    print("Please ensure the correct competition dataset is attached to your notebook.")
    sys.exit(1)

BASE_PATH = potential_base_paths[0]
TRAIN_LABELS_CSV = os.path.join(BASE_PATH, 'train_labels.csv')
TRAIN_DIR = os.path.join(BASE_PATH, 'train')

# Reasoning: Tries to find test data and submission template.
for dirname, subdirs, filenames in os.walk(BASE_PATH):
    if 'test_ids.csv' in filenames:
        TEST_IDS_CSV = os.path.join(dirname, 'test_ids.csv')
        if 'test' in subdirs:
            TEST_DIR = os.path.join(dirname, 'test')
        break
    if 'sample_submission.csv' in filenames and SUBMISSION_CSV_TEMPLATE is None:
        SUBMISSION_CSV_TEMPLATE = os.path.join(dirname, 'sample_submission.csv')


if not all([TRAIN_LABELS_CSV, TRAIN_DIR, TEST_DIR]):
    # Reasoning: Ensures all essential components for training and testing are found.
    print("CRITICAL ERROR: Could not find all necessary dataset components (train_labels.csv, train dir, test dir).")
    print(f"Discovered: TRAIN_LABELS_CSV={TRAIN_LABELS_CSV}, TRAIN_DIR={TRAIN_DIR}, TEST_DIR={TEST_DIR}")
    sys.exit(1)

print(f"Detected BASE_PATH: {BASE_PATH}")
print(f"Detected TRAIN_LABELS_CSV: {TRAIN_LABELS_CSV}")
print(f"Detected TRAIN_DIR: {TRAIN_DIR}")
print(f"Detected TEST_IDS_CSV: {TEST_IDS_CSV} (might be None if no test_ids.csv)")
print(f"Detected TEST_DIR: {TEST_DIR}")
print(f"Detected SUBMISSION_CSV_TEMPLATE: {SUBMISSION_CSV_TEMPLATE} (might be None if no sample_submission.csv)")

# Model and Data Configuration
IMAGE_SIZE = (128, 128) # Reasoning: Neural networks require fixed-size inputs. This standardizes all images.
BATCH_SIZE = 32 # Reasoning: Number of samples processed before updating model weights. Impacts memory usage and training speed.
EPOCHS = 50 # Reasoning: Number of complete passes through the training dataset. Autoencoders often benefit from more epochs to learn fine details.
LATENT_DIM = 128 # Reasoning: Dimensionality of the compressed representation (bottleneck) in the autoencoder. Tunable for balancing compression and information retention.
STDEV_MULTIPLIER = 2.0 # Reasoning: Hyperparameter for anomaly thresholding. Defines how many standard deviations above the mean error is considered anomalous.
SOIL_LABEL = 1 # Reasoning: Defines the positive class label (the 'normal' class that the autoencoder learns to reconstruct).
NOT_SOIL_LABEL = 0 # Reasoning: Defines the anomaly class label.

# --- 1. Load Training Data and Prepare DataFrame ---
# Reasoning: Reads image IDs and their corresponding labels from the CSV,
# then constructs full file paths. Essential for linking image files to their labels.
print("\n--- 1. Loading Training Data and Preparing DataFrame ---")
try:
    print(f"Attempting to read training CSV from: {TRAIN_LABELS_CSV}")

    train_df = pd.read_csv(
        TRAIN_LABELS_CSV,
        sep=',',
        header=0
    )

    train_df['label'] = pd.to_numeric(train_df['label'], errors='coerce') # Reasoning: Ensures labels are numbers, handles potential non-numeric entries.
    train_df.dropna(subset=['label'], inplace=True) # Reasoning: Removes rows with missing or invalid labels.

    train_df['image_path'] = train_df['image_id'].apply(lambda x: os.path.join(TRAIN_DIR, str(x))) # Reasoning: Creates full paths to image files.

    print(f"\nTraining DataFrame Shape after parsing: {train_df.shape}")
    print("Training DataFrame Head after parsing:")
    display(train_df.head()) # Kaggle utility to display DataFrames cleanly

    print("\n--- Diagnostic of train_df after parsing and cleaning ---")
    # Reasoning: These checks provide crucial insights into data integrity,
    # helping to identify issues like missing columns, unexpected label values,
    # or empty datasets that would break the model training.
    print(f"Is 'label' column present? {'label' in train_df.columns}")
    print(f"Is 'image_id' column present? {'image_id' in train_df.columns}")
    print(f"Unique values in 'label' column: {train_df['label'].unique()}")
    print(f"Value counts for 'label' column: \n{train_df['label'].value_counts()}")
    print(f"Number of NaN values in 'label' column: {train_df['label'].isnull().sum()}")
    print(f"dtype of 'label' column: {train_df['label'].dtype}")

    if train_df.empty:
        raise ValueError("train_df is empty after loading and cleaning. No valid labels or image IDs were found.")
    
    if len(train_df['label'].unique()) > 1:
        # Reasoning: Warns if multiple labels are present. An autoencoder is trained
        # on *one* class and detects deviations. If the problem is truly multi-class,
        # a standard classifier would be more appropriate.
        print("\nWARNING: Multiple unique labels found in train_labels.csv.")
        print("This autoencoder approach is primarily for one-class classification.")
        print("If you have multiple labels and the competition is multi-class, consider a standard CNN classifier.")
        print("Proceeding with autoencoder training on all data as if it's one class.")
    elif len(train_df['label'].unique()) == 1 and SOIL_LABEL not in train_df['label'].unique():
        # Reasoning: Ensures the single label found matches the expected SOIL_LABEL.
        print(f"\nCRITICAL ERROR: The only label found ({train_df['label'].unique()[0]}) does not match SOIL_LABEL ({SOIL_LABEL}).")
        sys.exit(1)
    else:
        print(f"\nAs expected, only one unique label ({train_df['label'].unique()[0]}) found. Proceeding with one-class autoencoder.")


except FileNotFoundError:
    print(f"\nCRITICAL ERROR: Training CSV file not found at '{TRAIN_LABELS_CSV}'.")
    print("Please ensure the dataset is correctly attached or the BASE_PATH is incorrect.")
    sys.exit(1)
except Exception as e:
    print(f"\nAn unexpected error occurred during initial data loading: {e}.")
    sys.exit(1)

# --- 2. Initial Exploratory Data Analysis (EDA) and Image Path/Dimension Check ---
# Reasoning: Basic checks to understand data structure, identify missing values,
# duplicates, and verify image file accessibility and dimensions. Crucial for data quality.
print("\n--- 2. Initial EDA and Image Path/Dimension Check ---")
print("\nDataFrame Info:")
train_df.info()
print("\nDataFrame Description:")
display(train_df.describe(include='all'))

print("\nMissing Values:")
print(train_df.isnull().sum())
print("\nNumber of duplicate rows:", train_df.duplicated().sum())

image_widths_eda = []
image_heights_eda = []
invalid_image_paths_eda = []

print("\nChecking image files and dimensions (this may take a moment)...")
for img_path in train_df['image_path']:
    try:
        img = Image.open(img_path).convert('RGB') # Reasoning: Attempts to open image to check for readability.
        width, height = img.size
        image_widths_eda.append(width)
        image_heights_eda.append(height)
    except FileNotFoundError:
        invalid_image_paths_eda.append(img_path)
    except (UnidentifiedImageError, Exception) as e:
        # Reasoning: Catches common image loading errors (e.g., corrupted files)
        # to identify problematic images.
        invalid_image_paths_eda.append(img_path)

if invalid_image_paths_eda:
    print(f"\nWARNING: {len(invalid_image_paths_eda)} invalid/unreadable images found.")
    print("These will be replaced with black placeholder images during model preprocessing.")
else:
    print("\nAll initial image path and readability checks passed successfully.")

if image_widths_eda and image_heights_eda:
    # Reasoning: Provides statistics on image dimensions to understand data consistency.
    print(f"Average image width: {np.mean(image_widths_eda):.2f}, min width: {min(image_widths_eda)}, max width: {max(image_widths_eda)}")
    print(f"Average image height: {np.mean(image_heights_eda):.2f}, min height: {min(image_heights_eda)}, max height: {max(image_heights_eda)}")
else:
    print("No valid image dimensions could be collected. This suggests issues with image paths or all images are corrupted.")

# --- 3. Analyze Label Distribution (simplified for one-class) ---
# Reasoning: Confirms the distribution of labels, reinforcing the one-class assumption for the autoencoder.
print("\n--- 3. Label Distribution (One-Class Focus) ---")
if not train_df.empty and 'label' in train_df.columns:
    label_counts = train_df['label'].value_counts()
    print(f"Training data contains {len(label_counts)} unique label(s).")
    print(f"Value counts: \n{label_counts}")
    print(f"Proceeding assuming all training data belongs to the 'soil' class ({SOIL_LABEL}).")
else:
    print("Cannot analyze label distribution: train_df is empty or 'label' column is missing.")


# --- 4. No explicit Label Encoding (labels are not used for training) ---
# Reasoning: Highlights that for an autoencoder, labels are not used for the training process itself
# (which is unsupervised). They are only used later for defining 'normal' vs. 'anomaly'.
print("\n--- 4. Label Encoding (N/A for Autoencoder training) ---")
print("Labels are not explicitly encoded for autoencoder training as it's unsupervised.")
print(f"The model will be trained to recognize the features of '{SOIL_LABEL}'.")


# --- 5. Image Loading and Preprocessing for Model Input ---
# Reasoning: This prepares the raw image data into a format suitable for the neural network.
print("\n--- 5. Loading and Preprocessing Images for Model ---")

def load_image_for_model(path, target_size=IMAGE_SIZE):
    try:
        img = Image.open(path).convert('RGB').resize(target_size) # Reasoning: Loads, converts to RGB (3 channels), and resizes for consistency.
        return np.array(img) / 255.0 # Reasoning: Normalizes pixel values to [0, 1]. Crucial for stable neural network training.
    except (FileNotFoundError, UnidentifiedImageError, Exception) as e:
        # Reasoning: Robust error handling to replace unreadable images with black placeholders,
        # preventing the script from crashing.
        print(f"Warning: Could not load image {path}. Replacing with black image. Error: {e}")
        return np.zeros((*target_size, 3), dtype=np.float32) # Reasoning: Returns a black image as a placeholder.

if not train_df.empty and 'image_path' in train_df.columns and not train_df['image_path'].empty:
    X = np.array([load_image_for_model(p) for p in train_df['image_path']]) # Reasoning: Loads all images into a single NumPy array for Keras.
    print(f"Shape of X (image features for autoencoder): {X.shape}")
else:
    print("Error: train_df is empty, 'image_path' column is missing, or no valid image paths exist.")
    sys.exit("Exiting: Cannot prepare data for model training.")

# --- 6. Train/Validation Split ---
# Reasoning: Splits the dataset. The autoencoder learns from X_train, and X_val is used to monitor
# its generalization performance during training, helping detect overfitting.
print("\n--- 6. Splitting Data into Training and Validation Sets ---")
X_train, X_val = train_test_split(X, test_size=0.2, random_state=42) # Reasoning: 80% for training, 20% for validation. Random state for reproducibility.
print(f"Training data shape (X_train): {X_train.shape}")
print(f"Validation data shape (X_val): {X_val.shape}")

# --- 7. Data Augmentation Setup ---
# Reasoning: Increases the diversity of the training data by applying random transformations.
# This helps the model generalize better to unseen images and reduces overfitting.
# For autoencoders, augmentations should not be too aggressive to ensure reconstructibility.
print("\n--- 7. Setting Up Data Augmentation (for Autoencoder) ---")
datagen = ImageDataGenerator(
    rotation_range=10, # Reasoning: Randomly rotates images by up to 10 degrees.
    width_shift_range=0.1, # Reasoning: Randomly shifts images horizontally by up to 10% of total width.
    height_shift_range=0.1, # Reasoning: Randomly shifts images vertically by up to 10% of total height.
    zoom_range=0.1, # Reasoning: Randomly zooms into images by up to 10%.
    horizontal_flip=True, # Reasoning: Randomly flips images horizontally.
    fill_mode='nearest' # Reasoning: Strategy for filling in new pixels created by transformations (e.g., shifts).
)
datagen.fit(X_train) # Reasoning: Computes internal data statistics for normalization/standardization if needed (though not explicitly used for normalization here due to prior /255.0).


# --- 8. Define and Compile Autoencoder Model ---
# Reasoning: This defines the core neural network architecture.
# An autoencoder learns to compress (encode) input data into a lower-dimensional
# latent space and then reconstruct (decode) it back to the original input.
# It's an unsupervised learning model, ideal for learning data distribution and anomaly detection.
print("\n--- 8. Defining and Compiling Autoencoder Model ---")

# Encoder: Compresses the input image into a lower-dimensional representation.
input_img = Input(shape=(*IMAGE_SIZE, 3)) # Reasoning: Defines the input layer with the expected image dimensions (height, width, channels).
x = Conv2D(32, (3, 3), activation='relu', padding='same')(input_img) # Reasoning: 32 filters, 3x3 kernel, ReLU activation. Extracts initial features.
x = MaxPooling2D((2, 2), padding='same')(x) # Reasoning: Downsamples (reduces spatial dimensions) by taking max value in each 2x2 window.
x = Conv2D(64, (3, 3), activation='relu', padding='same')(x) # Reasoning: More filters to learn more complex features after downsampling.
x = MaxPooling2D((2, 2), padding='same')(x)
x = Conv2D(128, (3, 3), activation='relu', padding='same')(x) # Reasoning: Even more filters for highly abstract features.
encoded = MaxPooling2D((2, 2), padding='same')(x) # Reasoning: Final downsampling to reach the most compressed (latent) representation.

# Decoder: Reconstructs the image from the compressed 'encoded' representation.
# It is designed to mirror the encoder's structure, but with upsampling.
x = Conv2D(128, (3, 3), activation='relu', padding='same')(encoded) # Reasoning: Starts reconstruction with 128 filters.
x = UpSampling2D((2, 2))(x) # Reasoning: Upsamples (increases spatial dimensions) by repeating rows/columns.
x = Conv2D(64, (3, 3), activation='relu', padding='same')(x) # Reasoning: Continues reconstruction with 64 filters.
x = UpSampling2D((2, 2))(x)
x = Conv2D(32, (3, 3), activation='relu', padding='same')(x) # Reasoning: Further reconstruction with 32 filters.
x = UpSampling2D((2, 2))(x)
# Final layer to output 3 channels (RGB) with pixel values between 0 and 1 (sigmoid)
decoded = Conv2D(3, (3, 3), activation='sigmoid', padding='same')(x) # Reasoning: Outputs an image with 3 channels (RGB). Sigmoid activation ensures pixel values are between 0 and 1 (normalized).

# Build the autoencoder model: Input is input_img, Output is decoded
autoencoder = Model(input_img, decoded) # Reasoning: Defines the end-to-end autoencoder model.
autoencoder.compile(optimizer='adam', loss='mse') # Reasoning: Adam optimizer for efficient training. MSE (Mean Squared Error) loss is standard for reconstruction tasks, aiming to minimize pixel-wise differences between input and output.

autoencoder.summary() # Reasoning: Prints a summary of the model's layers, output shapes, and number of parameters.


# --- 9. Model Training ---
# Reasoning: This is the learning phase where the autoencoder adjusts its weights
# to minimize the reconstruction error between input images and their decoded versions.
print("\n--- 9. Model Training ---")
history = autoencoder.fit(datagen.flow(X_train, X_train, batch_size=BATCH_SIZE), # Reasoning: Feeds augmented training data. Input and target are both X_train (unsupervised).
                            validation_data=(X_val, X_val), # Reasoning: Uses validation data to monitor performance on unseen data, detecting overfitting.
                            epochs=EPOCHS, # Reasoning: Number of passes through the entire training dataset.
                            verbose=1) # Reasoning: Displays training progress.

# --- 10. Plot Training History ---
# Reasoning: Visualizes the training and validation loss over epochs.
# This plot helps assess if the model is learning, converging, and if overfitting is occurring.
print("\n--- 10. Plotting Training History ---")
plt.figure(figsize=(10, 5))
plt.plot(history.history['loss'], label='Train Loss (MSE)')
plt.plot(history.history['val_loss'], label='Validation Loss (MSE)')
plt.title('Autoencoder Reconstruction Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss (MSE)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# --- 11. Calculate Reconstruction Errors and Set Threshold ---
# Reasoning: This is the core of the anomaly detection strategy.
# The autoencoder, trained only on 'normal' (soil) images, will have low
# reconstruction errors for similar normal images. Anomalies will have high errors.
print("\n--- 11. Calculating Reconstruction Errors and Setting Threshold ---")

# Calculate reconstruction errors for the training set
train_reconstructions = autoencoder.predict(X_train) # Reasoning: Get the reconstructed versions of the training images.
train_errors = np.mean(np.square(X_train - train_reconstructions), axis=(1, 2, 3)) # Reasoning: Calculate Mean Squared Error for each image. High error = poor reconstruction.

mean_train_error = np.mean(train_errors) # Reasoning: Average error for 'normal' training data.
std_train_error = np.std(train_errors) # Reasoning: Variability of errors for 'normal' data.

# Set the threshold based on mean + k * std_dev
# Reasoning: This statistical method sets a boundary. Images with errors beyond this
# boundary are considered anomalous. STDEV_MULTIPLIER controls sensitivity.
reconstruction_threshold = mean_train_error + STDEV_MULTIPLIER * std_train_error

print(f"Mean Reconstruction Error (Training): {mean_train_error:.6f}")
print(f"Standard Deviation of Reconstruction Error (Training): {std_train_error:.6f}")
print(f"Calculated Reconstruction Threshold (Mean + {STDEV_MULTIPLIER}*StdDev): {reconstruction_threshold:.6f}")

# Visualize the distribution of training errors and the threshold
# Reasoning: Helps to visually understand the spread of 'normal' errors and where the chosen threshold lies.
plt.figure(figsize=(10, 6))
plt.hist(train_errors, bins=50, density=True, alpha=0.6, color='g', label='Training Reconstruction Errors')
plt.axvline(reconstruction_threshold, color='r', linestyle='--', label=f'Threshold: {reconstruction_threshold:.4f}')
# Add Gaussian fit for visualization
xmin, xmax = plt.xlim()
x = np.linspace(xmin, xmax, 100)
p = norm.pdf(x, mean_train_error, std_train_error) # Reasoning: Plots a normal distribution curve to visualize the statistical fit.
plt.plot(x, p, 'k', linewidth=2, label='Normal Distribution Fit')
plt.title('Distribution of Reconstruction Errors on Training Data')
plt.xlabel('Reconstruction Error (MSE)')
plt.ylabel('Density')
plt.legend()
plt.grid(True)
plt.show()


# --- 12. Process Test Data and Generate Submission ---
# Reasoning: Loads, preprocesses, and classifies test images based on the learned threshold.
print("\n--- 12. Processing Test Data and Generating Submission ---")
try:
    if TEST_IDS_CSV:
        test_df = pd.read_csv(
            TEST_IDS_CSV,
            sep=',',
            header=0,
            on_bad_lines='warn' # Reasoning: Handles potentially malformed lines in CSV.
        )
        if 'image_id' not in test_df.columns:
            # Reasoning: Fallback if test_ids.csv is malformed or missing 'image_id'.
            # Directly infers IDs from filenames in the test directory.
            print("CRITICAL WARNING: 'image_id' column not found in test_ids.csv. Please check its structure.")
            test_image_ids = sorted([f for f in os.listdir(TEST_DIR) if f.endswith(('.jpg', '.png', '.jpeg'))])
            test_df = pd.DataFrame({'image_id': test_image_ids})

    else:
        # Reasoning: If no test_ids.csv is provided, assume all images in TEST_DIR are for testing.
        print("No test_ids.csv found. Inferring test image IDs directly from TEST_DIR.")
        test_image_ids = sorted([f for f in os.listdir(TEST_DIR) if f.endswith(('.jpg', '.png', '.jpeg'))])
        test_df = pd.DataFrame({'image_id': test_image_ids})

    test_df['image_path'] = test_df['image_id'].apply(lambda x: os.path.join(TEST_DIR, str(x))) # Reasoning: Construct full paths for test images.

    print(f"Test DataFrame Shape: {test_df.shape}")
    print("Test DataFrame Head:")
    display(test_df.head())

    missing_test_images = [path for path in test_df['image_path'] if not os.path.exists(path)] # Reasoning: Checks if test images actually exist.
    if missing_test_images:
        print(f"WARNING: {len(missing_test_images)} images listed in test_df are missing from {TEST_DIR}.")

except FileNotFoundError:
    print(f"\nCRITICAL ERROR: Test directory or test_ids.csv not found at '{TEST_DIR}' / '{TEST_IDS_CSV}'.")
    sys.exit(1)
except Exception as e:
    print(f"\nAn error occurred loading test data: {e}.")
    sys.exit(1)

X_test = np.array([load_image_for_model(p) for p in test_df['image_path']]) # Reasoning: Loads and preprocesses test images.
print(f"Shape of processed test images: {X_test.shape}")

if X_test.shape[0] > 0:
    test_reconstructions = autoencoder.predict(X_test) # Reasoning: Get reconstructions for test images.
    test_errors = np.mean(np.square(X_test - test_reconstructions), axis=(1, 2, 3)) # Reasoning: Calculate reconstruction error for each test image.

    # Classify based on the threshold
    # Reasoning: If an image's reconstruction error is above the threshold, it's an anomaly ('not soil'); otherwise, it's 'soil'.
    predicted_labels = np.where(test_errors > reconstruction_threshold, NOT_SOIL_LABEL, SOIL_LABEL)
    
    print("\nExample Test Image Reconstruction Errors and Predicted Labels:")
    for i in range(min(5, len(test_errors))): # Display first 5
        print(f"Image ID: {test_df.iloc[i]['image_id']}, Error: {test_errors[i]:.6f}, Predicted Label: {predicted_labels[i]}")

else:
    print("Skipping prediction: No test images to predict.")
    predicted_labels = [] # Ensure it's an empty list if not predicting

# --- 13. Generate Submission File ---
# Reasoning: Creates the final submission file in the format required by the competition.
print("\n--- 13. Generating Submission File ---")
if not test_df.empty and len(predicted_labels) > 0:
    submission = pd.DataFrame({
        'image_id': test_df['image_id'],
        'label': predicted_labels
    })

    print("\nSubmission Head:")
    display(submission.head())

    submission.to_csv('submission.csv', index=False) # Reasoning: Saves the DataFrame to a CSV file. index=False prevents writing the DataFrame index as a column.
    print("\nSubmission file 'submission.csv' generated successfully!")
else:
    print("Skipping submission file generation: Test data or predictions are missing.")


try:
    # Save the autoencoder model
    autoencoder.save("soil_autoencoder_model.h5") # Reasoning: Saves the trained model for future use without retraining.
    print("\nAutoencoder model saved as 'soil_autoencoder_model.h5'")
except Exception as e:
    print(f"\nWarning: Could not save the model. Error: {e}")