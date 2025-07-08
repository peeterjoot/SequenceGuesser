## Setup:

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
