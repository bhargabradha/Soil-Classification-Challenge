# ==============================================================================
# SECTION 1: IMPORTS AND GLOBAL CONFIGURATION
# Purpose: Import necessary libraries and set up global parameters for the competition.
# ==============================================================================
import os
import pandas as pd
from PIL import Image # For opening and processing images
from sklearn.model_selection import train_test_split # For splitting data into train/validation sets
from sklearn.metrics import f1_score # ADDED: For calculating F1-score during validation

import torch # PyTorch deep learning library
from torch.utils.data import Dataset, DataLoader # Tools for handling datasets and loading data in batches
import torchvision.transforms as transforms # For image transformations (resize, convert to tensor)
import torch.nn as nn # Neural network modules
import torch.optim as optim # Optimization algorithms (like Adam)
from torchvision.models import resnet18, ResNet18_Weights # Pre-trained ResNet18 model

# --- ADDED: Plotting Libraries ---
# Reason: These libraries are essential for creating and displaying the training graphs.
# matplotlib.pyplot is the core plotting library for creating static, animated, and interactive visualizations.
# seaborn makes the plots look aesthetically nicer with minimal effort and provides a high-level interface for drawing attractive statistical graphics.
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style("whitegrid") # Applies a pleasant grid style to the plots, improving readability.
# --- END ADDED ---


# ==============================================================================
# SECTION 2: CUSTOM DATASET CLASS
# Purpose: Define how your images and labels are loaded and processed for PyTorch.
# Reasoning: PyTorch requires data to be organized into a `Dataset` object. This custom class
#            inherits from `torch.utils.data.Dataset` and implements two key methods:
#            `__len__` (to return the total number of samples) and
#            `__getitem__` (to load a single sample given an index). This abstraction
#            allows `DataLoader` to efficiently handle batching and shuffling.
# ==============================================================================
class SoilDataset(Dataset):
    def __init__(self, dataframe, img_dir, is_train=True, transform=None):
        """
        Initializes the dataset.
        Args:
            dataframe (pd.DataFrame): DataFrame containing image IDs and (for train) labels.
            img_dir (str): Directory where image files are stored.
            is_train (bool): True for training/validation set, False for test set.
                             Reasoning: This flag allows the same Dataset class to handle
                             both labelled data (train/val) and unlabelled data (test),
                             returning either `(image, label)` or `(image, image_id)`.
            transform (torchvision.transforms.Compose): Image transformations to apply.
                             Reasoning: These transformations (defined in SECTION 3)
                             are applied to each image as it's loaded, ensuring consistent
                             preprocessing before being fed to the model.
        """
        self.data = dataframe.reset_index(drop=True) # Reset index to ensure clean access for iloc.
        self.img_dir = img_dir
        self.is_train = is_train
        self.transform = transform

    def __len__(self):
        """
        Returns the total number of items in the dataset.
        Reasoning: Required by PyTorch's `DataLoader` to know the size of the dataset.
        """
        return len(self.data)

    def __getitem__(self, idx):
        """
        Retrieves an item (image and its label/filename) by index.
        Args:
            idx (int): Index of the item to retrieve.
        Returns:
            tuple: (image, label) if is_train is True, or (image, filename) if is_train is False.
        Reasoning: This method is called by the `DataLoader` to fetch individual samples.
                   It handles opening the image, applying transforms, and returning the
                   correct corresponding label or filename based on `is_train`.
        """
        # The first column (index 0) of your dataframe is the image filename/ID.
        img_name = os.path.join(self.img_dir, self.data.iloc[idx, 0]) # Constructs the full path to the image file.
        image = Image.open(img_name).convert('RGB') # Open image using PIL and ensure it's in 3-channel RGB format.
                                                   # Reasoning: Most pre-trained CNNs (like ResNet) expect 3-channel RGB input.

        if self.transform:
            image = self.transform(image) # Apply defined transformations to the image.

        if self.is_train:
            # For this competition, the second column (index 1) is the label (0, 1, 2, or 3 for 4 soil types)
            label = int(self.data.iloc[idx, 1]) # Extract the label and cast to integer.
            return image, label
        else:
            # For the test set, we don't have labels, so we return the filename (image_id) instead.
            # This filename will be used later to create the submission CSV.
            return image, self.data.iloc[idx, 0] # Return filename (image_id) for test set.

# ==============================================================================
# SECTION 3: IMAGE TRANSFORMS
# Purpose: Define how images are preprocessed before being fed into the model.
# Reasoning: Neural networks require consistent input. These transformations ensure
#            all images are in a uniform format (size, tensor type, normalized pixel values)
#            that matches the expectations of the pre-trained ResNet18 model.
# ==============================================================================
transform = transforms.Compose([
    transforms.Resize((224, 224)), # Resize all images to 224x224 pixels.
                                   # Reasoning: This is a standard input size for many pre-trained CNNs
                                   # (like ResNet which was trained on ImageNet images of this size),
                                   # ensuring compatibility and consistent input dimensions.
    transforms.ToTensor(),         # Convert PIL Image to PyTorch Tensor.
                                   # Reasoning: PyTorch models operate on Tensors. This also
                                   # automatically scales pixel values from [0, 255] to [0, 1],
                                   # which is a common and stable input range for neural networks.
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) # ImageNet normalization.
                                   # Reasoning: These specific mean and standard deviation values are
                                   # derived from the ImageNet dataset, on which ResNet18 was pre-trained.
                                   # Normalizing our data with these values ensures that our input data
                                   # has a similar distribution to the data the model originally learned from,
                                   # which is crucial for effective transfer learning and faster convergence.
])

# ==============================================================================
# SECTION 4: LOAD AND SPLIT DATA
# Purpose: Load the main CSV and split it into training and validation sets.
# Reasoning: Splitting the data allows us to train the model on one portion
#            and evaluate its performance on an unseen portion (validation set),
#            providing an unbiased estimate of its generalization capability and
#            helping to detect overfitting.
# ==============================================================================
# Load the training labels CSV file. Assuming this is the correct file for the 4-class competition.
full_df = pd.read_csv('/kaggle/input/soil-classification-2/soil_competition-2025/train_labels.csv')

# Split the full dataset into training and validation sets.
# test_size=0.2 means 20% of data for validation, 80% for training.
# stratify=full_df['label'] ensures that the proportion of each class is the same in both train and val sets.
# Reasoning: This is crucial for multi-class classification, especially if classes are imbalanced.
#            It prevents a situation where, by random chance, one split gets too few samples of a class,
#            leading to biased training or unreliable validation metrics.
# random_state=42 ensures the split is reproducible (same split every time you run the code).
train_df, val_df = train_test_split(full_df, test_size=0.2, stratify=full_df['label'], random_state=42)

# ==============================================================================
# SECTION 5: IMAGE PATHS
# Purpose: Define the directories where training and test images are located.
# Reasoning: Explicitly defining these paths makes the code clear about data locations
#            and makes it easier to adapt the script if image directories change.
# ==============================================================================
img_path = '/kaggle/input/soil-classification-2/soil_competition-2025/train' # Path to training/validation images.
test_img_path = '/kaggle/input/soil-classification-2/soil_competition-2025/test' # Path to test images.

# ==============================================================================
# SECTION 6: DATALOADERS
# Purpose: Create DataLoader objects to efficiently load data in batches for training and validation.
# Reasoning: DataLoaders abstract away the complexities of data loading. They efficiently
#            provide data in mini-batches, handle shuffling, and can perform multi-threaded
#            data loading, which significantly speeds up the training process by
#            keeping the GPU supplied with data.
# ==============================================================================
train_loader = DataLoader(
    SoilDataset(train_df, img_path, is_train=True, transform=transform),
    batch_size=32, # Reasoning: Process 32 images at a time. This batch size is a common choice
                   # balancing memory usage and computational efficiency on GPUs.
    shuffle=True   # Reasoning: Shuffling the training data in each epoch is crucial
                   # to prevent the model from memorizing the order of samples and
                   # to improve generalization.
)
val_loader = DataLoader(
    SoilDataset(val_df, img_path, is_train=True, transform=transform),
    batch_size=32, # Reasoning: Consistent batch size for validation.
    shuffle=False  # Reasoning: For validation (and testing), shuffling is not necessary
                   # as we are just evaluating performance, and keeping the order fixed
                   # ensures reproducibility of evaluation metrics.
)

# ==============================================================================
# SECTION 7: MODEL SETUP
# Purpose: Define the neural network model, loss function, and optimizer.
# Reasoning: This section configures the core components that enable the model to learn.
# ==============================================================================
# Determine if a GPU (CUDA) is available, otherwise use CPU.
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Reasoning: Utilizing a GPU (CUDA) significantly accelerates deep learning computations
#            due to its parallel processing capabilities, drastically reducing training time.

# Load a pre-trained ResNet18 model.
weights = ResNet18_Weights.DEFAULT # Specifies to use the default ImageNet pre-trained weights.
model = resnet18(weights=weights)
# Reasoning: Transfer learning. ResNet18 is a powerful CNN architecture. By loading weights
#            pre-trained on the massive ImageNet dataset, the model already possesses
#            a strong ability to recognize general visual features (e.g., edges, textures).
#            This drastically reduces training time and often leads to higher accuracy
#            even with smaller, specific datasets like ours.

# --- MODIFIED: Output layer for 4 classes (original competition) ---
model.fc = nn.Linear(model.fc.in_features, 4) # Replaces the final classification layer.
# Reasoning: The original ResNet18's `fc` (fully connected) layer was designed to output
#            1000 classes (for ImageNet). We replace it with a new `nn.Linear` layer that
#            takes the same number of input features (`model.fc.in_features`) but produces
#            exactly `4` output features. These 4 outputs correspond to our 4 distinct soil types.
#            This adaptation makes the pre-trained feature extractor suitable for our specific task.
# --- END MODIFIED ---

# Move the model to the chosen device (GPU if available, otherwise CPU).
model = model.to(device)
# Reasoning: Ensures all model computations happen on the selected device, leveraging GPU acceleration if available.

# Define the loss function: CrossEntropyLoss is suitable for multi-class classification.
criterion = nn.CrossEntropyLoss()
# Reasoning: CrossEntropyLoss is the standard and most appropriate loss function for multi-class
#            classification. It effectively measures the difference between the model's predicted
#            class probabilities (logits) and the true class labels, guiding the model to minimize this error.

# Define the optimizer: Adam is a popular choice for deep learning models.
optimizer = optim.Adam(model.parameters(), lr=0.001)
# Reasoning: Adam (Adaptive Moment Estimation) is a robust and widely used optimization algorithm.
#            It adaptively adjusts the learning rate for each parameter, often leading to
#            faster convergence and better performance compared to simpler optimizers.
#            `model.parameters()` tells the optimizer which parameters (weights and biases) to update.
#            `lr=0.001` sets the initial learning rate, controlling the step size for weight updates.

# ==============================================================================
# SECTION 8: TRAINING AND VALIDATION LOOP
# Purpose: Train the model over multiple epochs and evaluate its performance on the validation set.
# Reasoning: This iterative process is the core of deep learning training. The model learns
#            by repeatedly processing data, calculating errors, and adjusting its weights.
#            Validation ensures the model generalizes well to unseen data.
# ==============================================================================
num_epochs = 10 # You can adjust this. If you got 1.0000, 10 might be enough.
# Reasoning: Defines how many full passes the model will make over the entire training dataset.
#            This hyperparameter is tuned to balance underfitting (too few epochs) and overfitting (too many).

# History tracking for graphs
history = {
    'train_loss': [],    # To store average training loss per epoch.
    'val_accuracy': [],  # To store validation accuracy per epoch.
    'val_f1': []         # To store validation F1-score per epoch.
}
# Reasoning: Collecting these metrics allows for visualization of training progress,
#            which is crucial for monitoring learning, detecting issues like overfitting,
#            and making decisions about model training.

for epoch in range(num_epochs): # Loop through each epoch
    # Set the model to training mode.
    model.train()
    # Reasoning: Activates behaviors specific to training (e.g., Dropout layers are active,
    #            BatchNorm layers use batch statistics for learning).
    train_loss = 0.0
    for inputs, labels in train_loader: # Iterate over batches of training data
        inputs, labels = inputs.to(device), labels.to(device) # Move data to GPU/CPU.
        # Reasoning: Data must be on the same device as the model for computation.

        optimizer.zero_grad() # Clear gradients from the previous batch.
        # Reasoning: Gradients accumulate by default in PyTorch. This step is crucial
        #            to prevent gradients from previous iterations from interfering with the
        #            current batch's gradient calculation.
        outputs = model(inputs) # Forward pass: get model predictions (raw scores/logits).
        loss = criterion(outputs, labels) # Calculate the loss (error) between predictions and true labels.
        loss.backward() # Backward pass: compute gradients of the loss with respect to all model parameters.
        # Reasoning: This is where backpropagation happens, calculating how much each parameter contributes to the error.
        optimizer.step() # Update model weights using the optimizer based on computed gradients.
        # Reasoning: This is the actual learning step where the model's parameters are adjusted
        #            to minimize the loss.
        train_loss += loss.item() # Accumulate training loss for the epoch. `.item()` extracts the scalar value.

    # Validation Phase
    # Set the model to evaluation mode.
    model.eval()
    # Reasoning: Deactivates behaviors specific to training (e.g., Dropout layers are off,
    #            BatchNorm layers use learned statistics instead of batch statistics for consistent inference).
    val_correct = 0
    val_total = 0
    val_preds_list = [] # To collect predictions for F1-score
    val_labels_list = [] # To collect true labels for F1-score

    with torch.no_grad(): # Disable gradient calculation during validation.
        # Reasoning: During evaluation, we don't need to compute or store gradients
        #            as we're not updating model weights. This saves memory and speeds up computation.
        for inputs, labels in val_loader: # Iterate over batches of validation data
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1) # Get the predicted class (index of max probability) for each sample.
            # Reasoning: `torch.max(outputs, 1)` returns both the maximum value and its index along dimension 1 (class dimension).
            #            The index is the predicted class.
            val_correct += (preds == labels).sum().item() # Count correct predictions.
            val_total += labels.size(0) # Accumulate total samples.

            val_preds_list.extend(preds.cpu().numpy()) # Collect predictions on CPU for metric calculation.
            val_labels_list.extend(labels.cpu().numpy()) # Collect true labels on CPU for metric calculation.

    val_acc = 100 * val_correct / val_total # Calculate accuracy as a percentage.
    # --- MODIFIED: F1-score for multi-class (4 classes) ---
    val_f1 = f1_score(val_labels_list, val_preds_list, average='weighted', zero_division=0) # Calculate weighted F1-score.
    # Reasoning: For multi-class classification, `average='weighted'` is appropriate. It calculates the F1-score
    #            for each class and then takes a weighted average based on the number of true instances for each label.
    #            This accounts for class imbalance. `zero_division=0` prevents errors if a class has no true instances.
    # --- END MODIFIED ---

    # Append metrics to history
    history['train_loss'].append(train_loss / len(train_loader)) # Average training loss for the epoch.
    history['val_accuracy'].append(val_acc)
    history['val_f1'].append(val_f1)

    # Print epoch summary
    print(f"Epoch {epoch + 1}/{num_epochs}, Train Loss: {train_loss:.4f}, Val Accuracy: {val_acc:.2f}%, Val F1: {val_f1:.4f}")

# ==============================================================================
# SECTION 9: SAVE TRAINED MODEL
# Purpose: Save the weights of the trained model so it can be reloaded later.
# Reasoning: Persisting the model's state (its learned parameters) allows you to
#            reuse the trained model for inference or further training without
#            needing to retrain it from scratch, saving significant time.
# ==============================================================================
torch.save(model.state_dict(), "resnet18_soil.pth")
# Reasoning: `model.state_dict()` saves only the learned parameters (weights and biases) of the model.
#            This is generally preferred over saving the entire model object as it's smaller,
#            more flexible, and less prone to breaking if the model's class definition changes slightly.
#            The file "resnet18_soil.pth" will be saved in the current working directory (e.g., /kaggle/working/).

# ==============================================================================
# SECTION 10: PLOTTING TRAINING HISTORY
# Purpose: Visualize the training and validation performance over epochs.
# Reasoning: Plots provide a clear visual representation of how the model learned.
#            They help in:
#            - Monitoring convergence (if loss is decreasing).
#            - Identifying overfitting (if training loss decreases but validation accuracy/F1 stalls or worsens).
#            - Understanding the overall learning trend.
# ==============================================================================
print("\n" + "="*60)
print("--- Plotting Training History ---")
print("="*60 + "\n")

# Create a figure to hold three subplots
plt.figure(figsize=(18, 6)) # Increased figure width for 3 plots to ensure clarity and avoid overlap.

# Subplot 1: Training Loss
plt.subplot(1, 3, 1) # (rows, columns, plot_number) -> 1 row, 3 columns, this is the 1st plot.
plt.plot(history['train_loss'], label='Train Loss', color='blue')
plt.title(f'Train Loss over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True) # Adds a grid for better readability of values.

# Subplot 2: Validation Accuracy
plt.subplot(1, 3, 2) # This selects the 2nd position.
plt.plot(history['val_accuracy'], label='Validation Accuracy', color='green')
plt.title(f'Validation Accuracy over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Accuracy (%)')
plt.legend()
plt.grid(True)

# Subplot 3: Validation F1-score
plt.subplot(1, 3, 3) # This selects the 3rd position.
plt.plot(history['val_f1'], label='Validation F1 Score', color='orange')
plt.title(f'Validation F1 Score over Epochs (Weighted)') # Updated title to reflect 'weighted' average.
plt.xlabel('Epoch')
plt.ylabel('F1 Score')
plt.legend()
plt.grid(True)

plt.tight_layout() # Adjusts subplot parameters for a tight layout, preventing labels/titles from overlapping.
plt.show() # Displays the current figure with all subplots.

print("\n--- Plots Generated ---")

# ==============================================================================
# SECTION 11: PREDICT ON TEST SET
# Purpose: Use the trained model to make predictions on the unseen test images.
# Reasoning: After training, the model needs to make predictions on the dataset
#            for which true labels are unknown (the competition's test set).
#            These predictions are then formatted for submission.
# ==============================================================================
# Load test images and predict. Paths are assumed correct for the first competition.
test_ids_df = pd.read_csv('/kaggle/input/soil-classification-2/soil_competition-2025/test_ids.csv')
# Reasoning: Loads the metadata (image IDs) for the test set.

test_loader = DataLoader(
    SoilDataset(test_ids_df, test_img_path, is_train=False, transform=transform),
    batch_size=32,
    shuffle=False
)
# Reasoning: Creates a DataLoader for the test set. `is_train=False` ensures that
#            the Dataset returns `(image, filename)` pairs instead of `(image, label)`.
#            Shuffling is off as order doesn't matter for inference and reproducibility is desired.

model.eval() # Set the model to evaluation mode.
# Reasoning: Crucial for inference. It turns off layers like Dropout and ensures BatchNorm
#            uses its learned statistics, leading to consistent and correct predictions.
predictions = [] # Will store predicted class indices (0, 1, 2, or 3)
image_ids = []   # Will store corresponding image IDs for submission.

with torch.no_grad(): # Disable gradient computation during prediction.
    # Reasoning: No training (weight updates) is happening, so gradients are not needed.
    #            Disabling them saves memory and speeds up the inference process.
    for inputs, filenames in test_loader:
        inputs = inputs.to(device) # Move input images to the active device.
        outputs = model(inputs)    # Forward pass: get model's raw predictions.
        _, preds = torch.max(outputs, 1) # Get the predicted class (index of max probability).
        # Reasoning: Extracts the most probable class index for each image in the batch.
        predictions.extend(preds.cpu().numpy()) # Collect predictions, moving them to CPU and converting to NumPy array.
        image_ids.extend(filenames) # Collect corresponding image filenames.

# ==============================================================================
# SECTION 12: SAVE SUBMISSION FILE
# Purpose: Create a CSV file in the format required by Kaggle for submission.
# Reasoning: Competitions require predictions to be in a specific format for evaluation.
#            This section structures the model's output into the required CSV file.
# ==============================================================================
# Column names for Kaggle submission: 'image_id' and 'soil_type'
submission = pd.DataFrame({
    'image_id': image_ids,      # Maps collected image IDs to the 'image_id' column.
    'soil_type': predictions    # Maps collected predictions to the 'soil_type' column.
})
submission.to_csv("submission.csv", index=False) # Save the DataFrame to a CSV file.
# Reasoning: `index=False` prevents Pandas from writing the DataFrame's index as a column
#            in the CSV, which is usually not required for competition submissions.
print("Submission file saved as submission.csv")