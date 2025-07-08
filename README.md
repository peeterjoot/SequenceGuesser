## Setup:

This is a Claude guided attempt to make a pytorch program that builds a model to attempt to guess recurrence relations of the form

```
f_k = alpha * f_{k-1} + beta * f_{k-2}
```

Example usage (after a first call to train the model)

```
./sequence_predictor.py --param 1,1,1,1
=== Improved PyTorch Sequence Predictor ===
Learning recurrence relations: f_k = α*f_{k-1} + β*f_{k-2}

Model loaded from 'model.pth'
Normalization parameters:
  X_mean: 3.997800350189209, X_std: 3.1453535556793213
  y_mean: 5.762340068817139, y_std: 2.592698574066162

Testing on your example sequence...
True sequence: [1.0, 1.0, 2.0, 3.0, 5.0, 8.0, 13.0, 21.0, 34.0, 55.0]
Predicted:    [1.0, 1.0, 7.8, 21.9, 78.1, 230.2, 569.9, 1343.8, 2745.9, 5254.3]
Relative Error (%): [0.0, 0.0, 290.2, 631.6, 1462.7, 2777.9, 4283.7, 6299.0, 7976.0, 9453.2]
```

Examples:
```
# Use default parameters (2,3,1,2)
./sequence_predictor.py

# Test Fibonacci-like sequence
./sequence_predictor.py --params 1,1,1,1

# Test with visualization
./sequence_predictor.py --params 1.5,2.5,0,1 --visualize

# Force retraining with custom parameters
./sequence_predictor.py --params 3,1,2,1 --force-retrain --epochs 3000

# Test without saving
./sequence_predictor.py --params 2,2,1,1 --no-save
```

### Fedora 42:

```
sudo dnf install python3-torch python3-numpy python3-matplotlib
```

### Or, w/ virtual env:

```
sudo dnf install python3-venv

python3 -m venv pytorch-env

source pytorch-env/bin/activate

pip install torch numpy matplotlib
```
