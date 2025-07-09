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
    """Generate a sequence following f_k = α * f_{k-1} + β * f_{k-2}"""
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

def train_model(epochs, logTransformation):
    # Generate training data - focus on similar patterns to your example
    print("Generating training data...")
    sequences = []

    if logTransformation:
        params = [
            (2, 3, 1, 2),      # Your exact example
            (2, 3, 0, 1),      # Same coefficients, different start
            (2, 3, 2, 1),      # Same coefficients, swapped start
            (2, 3, 1, 3),      # Same coefficients, different b
            (2, 2, 1, 2),      # Similar but β=2
            (3, 2, 1, 2),      # Similar but α=3
            (2, 4, 1, 2),      # Similar but β=4
            (1.8, 2.5, 1, 2),  # Slightly smaller coefficients
            (2.2, 2.8, 1, 2),  # Slightly different coefficients
            (2, 3, 2, 3),      # Same coefficients, larger start
        ]
    else:
        # Fibonacci like sequences, with different starting points.  Generated with Mathematica:
        # Table[{RandomReal[{0.8, 1.05}], RandomReal[{0.8, 1.05}], RandomReal[{0, 2}], RandomReal[{0, 2}]}, {i, 0, 10}]
        params = [
            (0.861291, 0.840455, 1.75046, 0.970936),
            (1.02614, 1.01894, 1.04782, 1.38452),
            (0.936709, 0.904756, 1.39031, 0.583327),
            (0.942149, 1.03861, 1.36753, 1.94342),
            (0.984499, 0.869777, 1.3727, 0.259391),
            (0.863729, 0.933664, 1.9683, 0.358662),
            (0.947734, 0.967587, 0.189643, 0.290601),
            (0.911811, 0.857173, 1.45565, 1.65682),
            (1.00224, 0.90819, 1.1876, 0.0339141),
            (1.00587, 0.952368, 0.969396, 0.560593),
            (0.918202, 0.873014, 0.938231, 1.47218),
        ]
        #params = [
        #    (0.8, 0.9, 1, 2),
        #    (0.8, 0.9, 0, 1),
        #    (0.8, 0.9, 2, 1),
        #    (0.8, 0.9, 1, 3),
        #    (0.8, 0.9, 1, 2),
        #    (1.05, 0.9, 1, 2),
        #    (0.8, 0.9, 1, 2),
        #    (0.7, 1.0, 1, 2),
        #    (0.8, 0.7, 1, 2),
        #    (0.8, 0.9, 2, 3),
        #    #(1, 1, 1, 1),      # Fibonacci
        #    #(1, 1, 0, 1),      # Fibonacci starting from 0,1
        #    (1, 1, 2, 3),      # Fibonacci starting from 2,3
        #    (1, 1, 1, 2),      # Fibonacci starting from 1,2
        #]

    for alpha, beta, a, b in params:
        seq = generate_sequence(alpha, beta, a, b, length=10)
        if len(seq) >= 5:  # Need at least 5 points for training
            sequences.append(seq)
            print(f"α={alpha}, β={beta}, a={a}, b={b}: {seq}")

    # Create dataset
    X, y = create_dataset(sequences)
    print(f"\nDataset created: {len(X)} training examples")

    # Use log transformation to handle large values
    if logTransformation:
        X_log = np.log(np.maximum(X, 1e-8))  # Avoid log(0)
        y_log = np.log(np.maximum(y, 1e-8))

        # Normalize in log space
        X_mean, X_std = X_log.mean(), X_log.std()
        y_mean, y_std = y_log.mean(), y_log.std()
    else:
        X_mean, X_std = X.mean(), X.std()
        y_mean, y_std = y.mean(), y.std()

    # Avoid division by zero
    if X_std == 0:
        X_std = 1.0
    if y_std == 0:
        y_std = 1.0

    if logTransformation:
        X_norm = (X_log - X_mean) / X_std
        y_norm = (y_log - y_mean) / y_std
    else:
        X_norm = (X - X_mean) / X_std
        y_norm = (y - y_mean) / y_std

        # Check for any remaining NaN or inf values
        X_norm = np.nan_to_num(X_norm, nan=0.0, posinf=1.0, neginf=-1.0)
        y_norm = np.nan_to_num(y_norm, nan=0.0, posinf=1.0, neginf=-1.0)

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

def test_model(model, X_mean, X_std, y_mean, y_std, alpha, beta, a, b, logTransformation):
    """Test the model on your specific example"""
    print(f"\nTesting on α={alpha}, β={beta}, a={a}, b={b}...")

    test_sequence = generate_sequence(alpha, beta, a, b, length=10)
    #print(f"True sequence: {test_sequence}")
    print("True sequence:\t\t" + "\t".join(f"{x:.1f}" for x in test_sequence))

    # Test predictions
    model.eval()
    with torch.no_grad():
        predictions = [a, b]  # Starting values

        for i in range(len(test_sequence) - 2):
            # Prepare input (last two values)
            input_vals = np.array([predictions[-2], predictions[-1]], dtype=np.float32)

            if logTransformation:
                # Transform to log space
                input_log = np.log(np.maximum(input_vals, 1e-8))
            else:
                # Normalize input (with safety checks)
                if X_std != 0:
                    input_norm = (input_vals - X_mean) / X_std
                else:
                    input_norm = input_vals - X_mean

            # Normalize
            if logTransformation:
                input_norm = (input_log - X_mean) / X_std
            else:
                input_norm = np.nan_to_num(input_norm, nan=0.0, posinf=1.0, neginf=-1.0)

            input_tensor = torch.FloatTensor(input_norm).unsqueeze(0)

            # Make prediction
            pred_norm = model(input_tensor)
            if logTransformation:
                pred_log = pred_norm.item() * y_std + y_mean
                pred = float(np.exp(pred_log))  # Convert to regular float
            else:
                pred = pred_norm.item() * y_std + y_mean

            predictions.append(pred)

    #print(f"Predicted:    {[round(float(x), 1) for x in predictions]}")
    print("Predicted:\t\t" + "\t".join(f"{x:.1f}" for x in predictions))

    # Calculate relative errors (better for exponentially growing sequences)
    relative_errors = []
    absolute_errors = []
    for true, pred in zip(test_sequence, predictions):
        absolute_errors.append(abs(float(true - pred)))
        if true != 0:
            rel_error = (abs(true - pred) / true) * 100
            relative_errors.append(float(rel_error))
        else:
            relative_errors.append(0.0)

    #print(f"Relative Error (%): {[round(x, 1) for x in relative_errors]}")
    print("Absolute Error (%):\t" + "\t".join(f"{x:.1f}" for x in absolute_errors))
    print("Relative Error (%):\t" + "\t".join(f"{x:.1f}" for x in relative_errors))

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
    """Parse comma-separated parameters α,β,a,b"""
    try:
        params = [float(x.strip()) for x in value.split(',')]
        if len(params) != 4:
            raise argparse.ArgumentTypeError(f"Expected 4 parameters (α,β,a,b), got {len(params)}")

        alpha, beta, a, b = params
        if alpha == 0 and beta == 0:
            raise argparse.ArgumentTypeError("Both α and β cannot be zero")

        return params

    except ValueError as e:
        raise argparse.ArgumentTypeError(f"Invalid parameters: {value} - {e}")

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Sequence Predictor')

    # Boolean flags
    parser.add_argument('--visualize', '-v', action='store_true',
                       help='Show training loss plot')
    parser.add_argument('--logtx', '-g', action='store_true', default=False,
                       help='Use Log transformation')
    parser.add_argument('--save-model', action='store_true', default=True,
                       help='Save the trained model (default: True)')
    parser.add_argument('--no-save', action='store_true',
                       help='Do not save the model')
    #parser.add_argument('--verbose', action='store_true',
    #                   help='Enable verbose output')
    parser.add_argument('--params', type=parse_params, default=[1, 1, 1, 1],
                   help='Sequence parameters: α,β,a,b (default: 1,1,1,1)\n' +
                        'Examples: --params 0.5,0.5,1,3 or --params "0.2, 0.8, 0, 1"')

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
        model, X_mean, X_std, y_mean, y_std, losses = train_model(args.epochs, args.logtx)
        print("\nTraining complete!")

    # Test the model
    alpha, beta, a, b = args.params
    test_model(model, X_mean, X_std, y_mean, y_std, alpha, beta, a, b, args.logtx)

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
