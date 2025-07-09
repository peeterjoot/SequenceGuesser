## Setup:

This is a Claude assisted attempt to make a pytorch program that builds a model to attempt to guess recurrence relations of the form

```
f_k = alpha * f_{k-1} + beta * f_{k-2}
```

The model doesn't actually do a great job at prediction, but for some inputs at least tracks the order of magnitude.

Example:

```
rm -f *.pth ; ./sequence_predictor.py
=== Improved PyTorch Sequence Predictor ===
Learning recurrence relations: f_k = α*f_{k-1} + β*f_{k-2}

Generating training data...
α=0.8, β=0.9, a=1, b=2: [1.0, 2.0, 2.5, 3.8, 5.29, 7.652, 10.8826, 15.592880000000001, 22.268644000000002, 31.8485072]
α=0.8, β=0.9, a=0, b=1: [0.0, 1.0, 0.8, 1.54, 1.9520000000000004, 2.9476000000000004, 4.114880000000001, 5.944744000000002, 8.459187200000002, 12.117619360000003]
α=0.8, β=0.9, a=2, b=1: [2.0, 1.0, 2.6, 2.98, 4.724, 6.4612, 9.420560000000002, 13.351528000000002, 19.159726400000004, 27.344156320000007]
α=0.8, β=0.9, a=1, b=3: [1.0, 3.0, 3.3000000000000003, 5.340000000000001, 7.242000000000001, 10.599600000000002, 14.997480000000003, 21.537624000000005, 30.727831200000008, 43.966126560000006]
α=0.8, β=0.9, a=1, b=2: [1.0, 2.0, 2.5, 3.8, 5.29, 7.652, 10.8826, 15.592880000000001, 22.268644000000002, 31.8485072]
α=1.05, β=0.9, a=1, b=2: [1.0, 2.0, 3.0, 4.95, 7.897500000000001, 12.747375000000002, 20.49249375, 32.98975593750001, 53.08248810937501, 85.42739285859376]
α=0.8, β=0.9, a=1, b=2: [1.0, 2.0, 2.5, 3.8, 5.29, 7.652, 10.8826, 15.592880000000001, 22.268644000000002, 31.8485072]
α=0.7, β=1.0, a=1, b=2: [1.0, 2.0, 2.4, 3.6799999999999997, 4.975999999999999, 7.163199999999999, 9.990239999999998, 14.156367999999997, 19.899697599999996, 28.086156319999994]
α=0.8, β=0.7, a=1, b=2: [1.0, 2.0, 2.3, 3.2399999999999998, 4.202, 5.6296, 7.44508, 9.896784, 13.1289832, 17.43093536]
α=0.8, β=0.9, a=2, b=3: [2.0, 3.0, 4.2, 6.0600000000000005, 8.628, 12.3564, 17.65032, 25.241016000000002, 36.0781008, 51.57939504000001]
α=1, β=1, a=2, b=3: [2.0, 3.0, 5.0, 8.0, 13.0, 21.0, 34.0, 55.0, 89.0, 144.0]
α=1, β=1, a=1, b=2: [1.0, 2.0, 3.0, 5.0, 8.0, 13.0, 21.0, 34.0, 55.0, 89.0]

Dataset created: 96 training examples
Training model...
Epoch [200/2000], Loss: 0.055467
Epoch [400/2000], Loss: 0.011188
Epoch [600/2000], Loss: 0.011338
Epoch [800/2000], Loss: 0.011086
Epoch [1000/2000], Loss: 0.007222
Epoch [1200/2000], Loss: 0.009295
Epoch [1400/2000], Loss: 0.008763
Epoch [1600/2000], Loss: 0.005374
Epoch [1800/2000], Loss: 0.002976
Epoch [2000/2000], Loss: 0.006536

Training complete!

Testing on α=1, β=1, a=1, b=1...
True sequence:		1.0	1.0	2.0	3.0	5.0	8.0	13.0	21.0	34.0	55.0
Predicted:		1.0	1.0	3.8	5.3	8.2	12.0	17.1	22.8	29.5	37.8
Relative Error (%):	0.0	0.0	88.0	75.8	64.0	50.1	31.5	8.4	13.3	31.2

Model saved as 'model.pth'
```

If using α,β > 1, use --logtx when training or running the model.

Examples:
```
./sequence_predictor.py --logtx --params 2,3,1,2

# Test Fibonacci-like sequence (actually now the default)
./sequence_predictor.py --params 1,1,1,1

# Test with visualization
./sequence_predictor.py --logtx --params 1.5,2.5,0,1 --visualize

# Force retraining with custom parameters
./sequence_predictor.py --logtx --params 3,1,2,1 --force-retrain --epochs 3000

# Test without saving
./sequence_predictor.py --logtx --params 2,2,1,1 --no-save

# Fibonacci behaviour, but different starting points:
./sequence_predictor.py --force --visual --epoch 10000
./sequence_predictor.py  --param 1,1,0,1
./sequence_predictor.py  --param 1,1,0,2
./sequence_predictor.py  --param 1,1,1,3
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
