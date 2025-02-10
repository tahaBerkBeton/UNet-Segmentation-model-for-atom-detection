import os
import torch
import json
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from train import GridNetU
from tqdm import tqdm

# Determine the project root directory
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

# Load validation filenames using relative path
validation_json = os.path.join(ROOT_DIR, "validation_filenames.json")
with open(validation_json, 'r') as f:
    val_files = json.load(f)

# Load the model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = GridNetU().to(device)
checkpoint_path = os.path.join(ROOT_DIR, "best_model.pth")
checkpoint = torch.load(checkpoint_path, map_location=device)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# Define 10 threshold values evenly spaced between 0 and 1
thresholds = np.linspace(0, 1, 10)

# Lists to store the ROC curve data points
roc_tpr = []  # True Positive Rate (Sensitivity)
roc_fpr = []  # False Positive Rate (1 - Specificity)

# Evaluate the model for each threshold
for threshold in thresholds:
    total_tp, total_fp, total_fn, total_tn = 0, 0, 0, 0

    # Process all validation files for the current threshold
    for selected_file in tqdm(val_files, desc=f"Threshold {threshold:.2f}"):
        # Derive the corresponding truth filename
        truth_file = selected_file.replace('data_', 'truth_')

        # Load the data and ground truth images
        data_img = np.array(Image.open(selected_file)).astype(np.float32)
        truth_img = np.array(Image.open(truth_file)).astype(np.float32)

        # Normalize data (as in your training code)
        data_normalized = data_img / 265.0
        data_tensor = torch.from_numpy(data_normalized).unsqueeze(0).unsqueeze(0).to(device)

        # Get model prediction and apply sigmoid activation
        with torch.no_grad():
            prediction = torch.sigmoid(model(data_tensor))
        prediction = prediction.cpu().numpy().squeeze()

        # Binarize the prediction using the current threshold
        prediction_binary = (prediction > threshold).astype(np.float32)

        # Calculate confusion matrix components for this image
        tn, fp, fn, tp = confusion_matrix(
            truth_img.flatten(), 
            prediction_binary.flatten()
        ).ravel()

        # Accumulate values over the entire dataset
        total_tp += tp
        total_fp += fp
        total_fn += fn
        total_tn += tn

    # Compute TPR and FPR for the current threshold
    tpr = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
    fpr = total_fp / (total_fp + total_tn) if (total_fp + total_tn) > 0 else 0

    roc_tpr.append(tpr)
    roc_fpr.append(fpr)
    print(f"Threshold: {threshold:.2f} | TPR: {tpr:.4f} | FPR: {fpr:.4f}")

# Plot the ROC Curve
plt.figure(figsize=(8, 6))
plt.plot(roc_fpr, roc_tpr, marker='o', linestyle='-', color='b', label='ROC Curve')
# Annotate each point with its threshold value
for i, thr in enumerate(thresholds):
    plt.annotate(f"{thr:.2f}", (roc_fpr[i], roc_tpr[i]))
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve for Detection Thresholds")
plt.legend()
plt.grid(True)
roc_curve_path = os.path.join(ROOT_DIR, "roc_curve.png")
plt.savefig(roc_curve_path)
plt.show()

print(f"ROC curve saved to {roc_curve_path}")
