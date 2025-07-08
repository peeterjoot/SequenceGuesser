#! /usr/bin/python3

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

# Set random seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)

class SequencePredictor(nn.Module):
    def __init__(self, input_size=2, hidden_size=16, output_size=1):
        super(SequencePredictor, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x

def generate_sequence(alpha, beta, a, b, length=20):
    """Generate a sequence following f_k = alpha * f_{k-1} + beta * f_{k-2}"""
    sequence = [float(a), float(b)]
    for k in range(2, length):
        next_val = alpha * sequence[k-1] + beta * sequence[k-2]
        # Check for overflow
        if abs(next_val) > 1e6:  # Stop if values get too large
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

def train_model():
    # Generate training data with different parameters
    print("Generating training data...")
    sequences = []
    
    # Create multiple sequences with different parameters
    params = [
        (2, 3, 1, 2),      # Your example
        (1, 1, 1, 1),      # Fibonacci-like
        (1.5, -0.5, 2, 1), # Smaller coefficients
        (0.8, 0.3, 1, 3),  # Even smaller
        (1.2, 0.1, 0, 1),  # Starting from 0
    ]
    
    for alpha, beta, a, b in params:
        seq = generate_sequence(alpha, beta, a, b, length=15)  # Shorter sequences
        if len(seq) > 5:  # Only use if we got enough data points
            sequences.append(seq)
    
    # Create dataset
    X, y = create_dataset(sequences)
    
    # Normalize data for better training (with safety checks)
    X_mean, X_std = X.mean(), X.std()
    y_mean, y_std = y.mean(), y.std()
    
    # Avoid division by zero
    if X_std == 0:
        X_std = 1.0
    if y_std == 0:
        y_std = 1.0
    
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
    epochs = 1000
    
    for epoch in range(epochs):
        # Forward pass
        predictions = model(X_tensor)
        loss = criterion(predictions, y_tensor)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        losses.append(loss.item())
        
        if (epoch + 1) % 100 == 0:
            print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.6f}')
    
    return model, X_mean, X_std, y_mean, y_std, losses

def test_model(model, X_mean, X_std, y_mean, y_std):
    """Test the model on your specific example"""
    print("\nTesting on your example sequence...")
    
    # Your example: alpha=2, beta=3, a=1, b=2 (but shorter sequence)
    test_sequence = generate_sequence(2, 3, 1, 2, length=12)
    print(f"True sequence: {test_sequence}")
    
    # Test predictions
    model.eval()
    with torch.no_grad():
        predictions = [1.0, 2.0]  # Starting values
        
        for i in range(len(test_sequence) - 2):  # Predict remaining values
            # Prepare input (last two values)
            input_vals = np.array([predictions[-2], predictions[-1]], dtype=np.float32)
            
            # Normalize input (with safety checks)
            if X_std != 0:
                input_norm = (input_vals - X_mean) / X_std
            else:
                input_norm = input_vals - X_mean
            
            input_norm = np.nan_to_num(input_norm, nan=0.0, posinf=1.0, neginf=-1.0)
            input_tensor = torch.FloatTensor(input_norm).unsqueeze(0)
            
            # Make prediction
            pred_norm = model(input_tensor)
            pred = pred_norm.item() * y_std + y_mean
            
            # Check for numerical issues
            if np.isnan(pred) or np.isinf(pred):
                print(f"Warning: Numerical issue at step {i+3}")
                break
                
            predictions.append(pred)
    
    print(f"Predicted:    {[round(x, 1) for x in predictions]}")
    print(f"Error:        {[abs(true - pred) for true, pred in zip(test_sequence, predictions)]}")

def visualize_training(losses):
    """Plot training loss"""
    plt.figure(figsize=(10, 6))
    plt.plot(losses)
    plt.title('Training Loss Over Time')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.yscale('log')
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    print("=== Simple PyTorch Sequence Predictor ===")
    print("Learning recurrence relations: f_k = α*f_{k-1} + β*f_{k-2}")
    print()
    
    # Train the model
    model, X_mean, X_std, y_mean, y_std, losses = train_model()
    
    # Test the model
    test_model(model, X_mean, X_std, y_mean, y_std)
    
    # Visualize training (optional)
    print("\nTraining complete! Uncomment the line below to see training loss plot:")
    # visualize_training(losses)
    
    # Save the model
    torch.save({
        'model_state_dict': model.state_dict(),
        'X_mean': X_mean,
        'X_std': X_std,
        'y_mean': y_mean,
        'y_std': y_std
    }, 'sequence_predictor.pth')
    print("\nModel saved as 'sequence_predictor.pth'")
