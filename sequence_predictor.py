#! /usr/bin/python3

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import argparse
import sys
import os
import os.path
from pathlib import Path
import warnings

# Set random seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)

class SequencePredictor(nn.Module):
    def __init__(self, input_size=2, hidden_size=32, output_size=1):
        super(SequencePredictor, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, hidden_size)
        self.fc4 = nn.Linear(hidden_size, output_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.relu(self.fc3(x))
        x = self.fc4(x)
        return x

def generate_sequence(alpha, beta, a, b, length=12):
    """Generate a sequence following f_k = alpha * f_{k-1} + beta * f_{k-2}"""
    sequence = [float(a), float(b)]
    for k in range(2, length):
        next_val = alpha * sequence[k-1] + beta * sequence[k-2]
        # Check for overflow
        if abs(next_val) > 1e5:  # Lower threshold
            break
        sequence.append(next_val)
    return sequence

def create_dataset(sequences, window_size=2):
    """Create training dataset from sequences"""
    X, y = [], []
    for seq in sequences:
        for i in range(len(seq) - window_size):
            X.append(seq[i:i+window_size])
            y.append(seq[i+window_size])
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)

def train_model(epochs):
    # Generate training data - focus on similar patterns to your example
    print("Generating training data...")
    sequences = []

    # Create sequences similar to your example but with varied parameters
    params = [
        (2, 3, 1, 2),      # Your exact example
        (2, 3, 0, 1),      # Same coefficients, different start
        (2, 3, 2, 1),      # Same coefficients, swapped start
        (2, 3, 1, 3),      # Same coefficients, different b
        (2, 2, 1, 2),      # Similar but beta=2
        (3, 2, 1, 2),      # Similar but alpha=3
        (2, 4, 1, 2),      # Similar but beta=4
        (1.8, 2.5, 1, 2),  # Slightly smaller coefficients
        (2.2, 2.8, 1, 2),  # Slightly different coefficients
        (2, 3, 2, 3),      # Same coefficients, larger start
    ]

    for alpha, beta, a, b in params:
        seq = generate_sequence(alpha, beta, a, b, length=10)
        if len(seq) >= 5:  # Need at least 5 points for training
            sequences.append(seq)
            print(f"α={alpha}, β={beta}: {seq}")

    # Create dataset
    X, y = create_dataset(sequences)
    print(f"\nDataset created: {len(X)} training examples")

    # Use log transformation to handle large values
    X_log = np.log(np.maximum(X, 1e-8))  # Avoid log(0)
    y_log = np.log(np.maximum(y, 1e-8))

    # Normalize in log space
    X_mean, X_std = X_log.mean(), X_log.std()
    y_mean, y_std = y_log.mean(), y_log.std()

    # Avoid division by zero
    if X_std == 0:
        X_std = 1.0
    if y_std == 0:
        y_std = 1.0

    X_norm = (X_log - X_mean) / X_std
    y_norm = (y_log - y_mean) / y_std

    # Convert to PyTorch tensors
    X_tensor = torch.FloatTensor(X_norm)
    y_tensor = torch.FloatTensor(y_norm).unsqueeze(1)

    # Create model
    model = SequencePredictor()
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Training loop
    print("Training model...")
    losses = []

    for epoch in range(epochs):
        # Forward pass
        predictions = model(X_tensor)
        loss = criterion(predictions, y_tensor)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        losses.append(loss.item())

        if (epoch + 1) % 200 == 0:
            print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.6f}')

    return model, X_mean, X_std, y_mean, y_std, losses

def test_model(model, X_mean, X_std, y_mean, y_std, alpha, beta, a, b):
    """Test the model on your specific example"""
    print("\nTesting on your example sequence...")

    # Your example: alpha=2, beta=3, a=1, b=2
    test_sequence = generate_sequence(alpha, beta, a, b, length=10)
    print(f"True sequence: {test_sequence}")

    # Test predictions
    model.eval()
    with torch.no_grad():
        predictions = [a, b]  # Starting values

        for i in range(len(test_sequence) - 2):
            # Prepare input (last two values)
            input_vals = np.array([predictions[-2], predictions[-1]], dtype=np.float32)

            # Transform to log space
            input_log = np.log(np.maximum(input_vals, 1e-8))

            # Normalize
            input_norm = (input_log - X_mean) / X_std
            input_tensor = torch.FloatTensor(input_norm).unsqueeze(0)

            # Make prediction
            pred_norm = model(input_tensor)
            pred_log = pred_norm.item() * y_std + y_mean
            pred = float(np.exp(pred_log))  # Convert to regular float

            predictions.append(pred)

    print(f"Predicted:    {[round(float(x), 1) for x in predictions]}")

    # Calculate relative errors (better for exponentially growing sequences)
    relative_errors = []
    for true, pred in zip(test_sequence, predictions):
        if true != 0:
            rel_error = abs(true - pred) / true * 100
            relative_errors.append(float(rel_error))
        else:
            relative_errors.append(0.0)

    print(f"Relative Error (%): {[round(x, 1) for x in relative_errors]}")

def visualize_training(losses):
    """Plot training loss"""
    try:
        plt.figure(figsize=(10, 6))
        plt.plot(losses)
        plt.title('Training Loss Over Time')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.yscale('log')
        plt.grid(True)
        plt.show()
    except Exception as e:
        print(f"Could not display plot: {e}")

def load_model(model_file):
    """Load the saved model and normalization parameters"""

    # Load the saved data
    warnings.filterwarnings("ignore", category=FutureWarning, message=".*weights_only.*")
    checkpoint = torch.load(model_file, map_location=torch.device('cpu')) #, weights_only=True)

    # Create model instance
    model = SequencePredictor()

    # Load the model weights
    model.load_state_dict(checkpoint['model_state_dict'])

    # Extract normalization parameters
    X_mean = checkpoint['X_mean']
    X_std = checkpoint['X_std']
    y_mean = checkpoint['y_mean']
    y_std = checkpoint['y_std']

    # Set model to evaluation mode
    model.eval()

    print(f"Model loaded from '{model_file}'")
    print(f"Normalization parameters:")
    print(f"  X_mean: {X_mean}, X_std: {X_std}")
    print(f"  y_mean: {y_mean}, y_std: {y_std}")

    return model, X_mean, X_std, y_mean, y_std

def parse_params(value):
    """Parse comma-separated parameters alpha,beta,a,b"""
    try:
        params = [float(x.strip()) for x in value.split(',')]
        if len(params) != 4:
            raise argparse.ArgumentTypeError(f"Expected 4 parameters (alpha,beta,a,b), got {len(params)}")

        if alpha == 0 and beta == 0:
            raise argparse.ArgumentTypeError("Both alpha and beta cannot be zero")

        return params

    except ValueError as e:
        raise argparse.ArgumentTypeError(f"Invalid parameters: {value} - {e}")

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Sequence Predictor')

    # Boolean flags
    parser.add_argument('--visualize', '-v', action='store_true',
                       help='Show training loss plot')
    parser.add_argument('--save-model', action='store_true', default=True,
                       help='Save the trained model (default: True)')
    parser.add_argument('--no-save', action='store_true',
                       help='Do not save the model')
    #parser.add_argument('--verbose', action='store_true',
    #                   help='Enable verbose output')
    parser.add_argument('--params', type=parse_params, default=[2, 3, 1, 2],
                   help='Sequence parameters: alpha,beta,a,b (default: 2,3,1,2)\n' +
                        'Examples: --params 1.5,2.5,1,3 or --params "2.2, 3.8, 0, 1"')

    # Other parameters
    parser.add_argument('--epochs', type=int, default=2000,
                       help='Number of training epochs')
    parser.add_argument('--model-file', type=str, default='model.pth',
                       help='Model file path')

    parser.add_argument('--list-examples', action='store_true',
                   help='Show example parameter combinations')

    parser.add_argument('--force-retrain', action='store_true',
                       help='Force retraining even if model file exists')

    args = parser.parse_args()

    # Handle conflicting options
    if args.no_save:
        args.save_model = False

    print("=== Improved PyTorch Sequence Predictor ===")
    print("Learning recurrence relations: f_k = α*f_{k-1} + β*f_{k-2}")
    print()

    if args.list_examples:
        print("Interesting parameter combinations:")
        print("  Fibonacci-like: --params 1,1,1,1")
        print("  Exponential growth: --params 2,3,1,2")
        print("  Oscillating: --params 1,-1,1,2")
        print("  Geometric-like: --params 2,0,1,2")
        sys.exit(0)

    losses = None

    if os.path.isfile(args.model_file) and not args.force_retrain:
        model, X_mean, X_std, y_mean, y_std = load_model(args.model_file)
    else:
        # Train the model
        model, X_mean, X_std, y_mean, y_std, losses = train_model(args.epochs)
        print("\nTraining complete!")

    # Test the model
    alpha, beta, a, b = args.params
    test_model(model, X_mean, X_std, y_mean, y_std, alpha, beta, a, b)

    if losses is not None:
        # Visualize training (optional)
        if args.visualize:
            visualize_training(losses)

        # Save the model
        if args.save_model:
            torch.save({
                'model_state_dict': model.state_dict(),
                'X_mean': X_mean,
                'X_std': X_std,
                'y_mean': y_mean,
                'y_std': y_std
            }, args.model_file)
            print(f"\nModel saved as '{args.model_file}'")
