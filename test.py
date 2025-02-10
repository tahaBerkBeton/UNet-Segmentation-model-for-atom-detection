import os
import torch
import json
import numpy as np
from PIL import Image
import random
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix
from train import GridNetU

# Determine the project root directory
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

# Load validation filenames using relative path
validation_json = os.path.join(ROOT_DIR, "validation_filenames.json")
with open(validation_json, 'r') as f:
    val_files = json.load(f)

# Randomly select one validation file
selected_file = random.choice(val_files)
print(f"Selected file: {selected_file}")

# Load corresponding truth file (replace 'data_' with 'truth_' in filename)
truth_file = selected_file.replace('data_', 'truth_')
print(f"Corresponding truth file: {truth_file}")

# Load the model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = GridNetU().to(device)
checkpoint_path = os.path.join(ROOT_DIR, "best_model.pth")
checkpoint = torch.load(checkpoint_path, map_location=device)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

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

# Calculate metrics
precision, recall, f1, _ = precision_recall_fscore_support(
    truth_img.flatten(), 
    prediction_binary.flatten(), 
    average='binary'
)
tn, fp, fn, tp = confusion_matrix(truth_img.flatten(), prediction_binary.flatten()).ravel()

# Calculate additional metrics
accuracy = (tp + tn) / (tp + tn + fp + fn)
specificity = tn / (tn + fp)

print("\nModel Performance Metrics:")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1 Score: {f1:.4f}")
print(f"Accuracy: {accuracy:.4f}")
print(f"Specificity: {specificity:.4f}")

# Identify non-conforming spots (errors)
error_mask = prediction_binary != truth_img
error_indices = np.argwhere(error_mask)

# Plot the results
fig, axes = plt.subplots(1, 4, figsize=(20, 5))

axes[0].imshow(data_img, cmap='gray')
axes[0].set_title('Input Data')
axes[0].axis('off')

axes[1].imshow(truth_img, cmap='gray')
axes[1].set_title('Ground Truth')
axes[1].axis('off')

axes[2].imshow(prediction, cmap='gray')
axes[2].set_title('Prediction (Raw)')
axes[2].axis('off')

# Add circles around errors on the binary prediction
axes[3].imshow(prediction_binary, cmap='gray')
axes[3].set_title('Prediction (Binary) with Errors')
for y, x in error_indices:
    circle = Circle((x, y), radius=5, color='red', fill=False, linewidth=1)
    axes[3].add_patch(circle)
axes[3].axis('off')

plt.tight_layout()
eval_plot_path = os.path.join(ROOT_DIR, "evaluation_plot_with_errors.png")
plt.savefig(eval_plot_path)
plt.show()

# Store the images and metrics for visualization
results = {
    'data': data_img.tolist(),  
    'truth': truth_img.tolist(),
    'prediction': prediction.tolist(),
    'prediction_binary': prediction_binary.tolist(),
    'metrics': {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'accuracy': accuracy,
        'specificity': specificity
    }
}

# Only saving the metrics as JSON for brevity
results_json = {'metrics': results['metrics']}
print("\nDetailed results:", json.dumps(results_json, indent=2))
