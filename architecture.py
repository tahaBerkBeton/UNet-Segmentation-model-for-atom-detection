import os
import torch
from train import GridNetU  # Import your model from train.py
from torchviz import make_dot  # Library for visualizing PyTorch graphs

# Set the device (optional, but useful if you want to see GPU computations)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Initialize the model and move it to the device
model = GridNetU().to(device)
model.eval()  # Set to evaluation mode

# Create a dummy input tensor matching the input shape of your model
# Here, batch_size=1, channels=1, height=416, width=416 (as used in your training code)
dummy_input = torch.zeros(1, 1, 416, 416).to(device)

# Forward pass through the model to get the output (this is required by torchviz)
output = model(dummy_input)

# Create the graph visualization. 
# We pass model.named_parameters() to include parameter names in the graph.
graph = make_dot(output, params=dict(model.named_parameters()))

# Define output path (in the same directory as this script)
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
output_filename = os.path.join(ROOT_DIR, "GridNetU_architecture")

# Render the graph to a PDF file (alternatively, you can set format="png")
graph.render(output_filename, format="png")

print(f"Model architecture visualization saved as {output_filename}.png")
