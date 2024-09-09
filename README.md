# Protecting Label Distribution in Cross-Silo Federated Learning


This repository contains the PyTorch implementation of the LDPM mechanism proposed in the [Oakland'24 Paper](https://ieeexplore.ieee.org/document/10646748).

## Requirements 
- torch, torchvision
- kymatio
- numpy
- scipy
- jupyter
- autodp (https://github.com/yuxiangw/autodp)

## Files
> FLModel_row.py: Core component of the federated learning framework and LDPM.

> MLModel.py: Models for classification tasks (Logistic Regression paired with Scattering Networks).

> calibrating_iterations.ipyn: Calibrates the number of SGD iterations based on our closed-form privacy guarantee bound.

## Usage
### Step 1: Generate non-i.i.d. local datasets
We use Dirichlet distribution for data splitting. Run Jupyter notebook ```(fashion-)mnist-noniid.ipynb``` to generate local datasets with non-i.i.d. label distributions.

### Step 2: Calibrate noise/number of SGD iterations
Our closed-form privacy bounds are implemented in ```calibrating_iterations.ipynb```. The function ```get_sampled_row_rdp``` returns the Renyi label distributional privacy guarantee $\kappa$.

Key parameters in ```calibrating_iterations.ipynb```:
```python
# code snippet from calibrating_iterations.ipynb
bad_event = 1e-5  # privacy budget $\xi$
sigma = 10        # std of Gaussian noise
m = 32            # group size
c = 10            # number of classes
q = 0.05          # Poisson sampling rate
```

### Step 3: Setup parameters for FL training
Key parameters for FL framework:
```python
# code snippet from train_scatter_mnist.ipynb
lr = 0.1
fl_param = {
    'output_size': 10,          # number of units in output layer
    'client_num': client_num,   # number of clients
    'model': 'scatter', # model type
    'data': d,          # dataset
    'lr': lr,           # learning rate
    'E': 100,           # number of local iterations
    'q': 0.05,          # sampling rate
    'clip': 1.0,        # l2 clipping norm 
    'noise': 10.0,      # gaussian noise std
    'clip': 1,          # clipping norm
    'tot_T': 5,         # number of global iterations
}
```