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

# Initialize metrics accumulators
total_tp, total_fp, total_fn, total_tn = 0, 0, 0, 0

# Process all validation files with a progress bar
for selected_file in tqdm(val_files, desc="Processing validation files"):
    # Load corresponding truth file (replace 'data_' with 'truth_' in filename)
    truth_file = selected_file.replace('data_', 'truth_')

    # Load and preprocess the data
    data_img = np.array(Image.open(selected_file)).astype(np.float32)
    truth_img = np.array(Image.open(truth_file)).astype(np.float32)

    # Normalize data
    data_normalized = data_img / 265.0
    data_tensor = torch.from_numpy(data_normalized).unsqueeze(0).unsqueeze(0).to(device)

    # Get model prediction
    with torch.no_grad():
        prediction = torch.sigmoid(model(data_tensor))
        prediction = prediction.cpu().numpy().squeeze()

    # Convert prediction to binary (threshold = 0.5)
    prediction_binary = (prediction > 0.5).astype(np.float32)

    # Calculate confusion matrix components
    tn, fp, fn, tp = confusion_matrix(
        truth_img.flatten(), 
        prediction_binary.flatten()
    ).ravel()

    # Accumulate metrics
    total_tp += tp
    total_fp += fp
    total_fn += fn
    total_tn += tn

# Calculate overall metrics
precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
accuracy = (total_tp + total_tn) / (total_tp + total_fp + total_fn + total_tn)
specificity = total_tn / (total_tn + total_fp) if (total_tn + total_fp) > 0 else 0

print("\nOverall Model Performance on Validation Dataset:")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1 Score: {f1:.4f}")
print(f"Accuracy: {accuracy:.4f}")
print(f"Specificity: {specificity:.4f}")

# Store overall metrics for visualization
overall_results = {
    'metrics': {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'accuracy': accuracy,
        'specificity': specificity
    }
}

# Save metrics to JSON using a relative path
metrics_path = os.path.join(ROOT_DIR, "overall_metrics.json")
with open(metrics_path, 'w') as f:
    json.dump(overall_results, f, indent=4)
print(f"\nOverall metrics saved to {metrics_path}")
