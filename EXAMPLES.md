# Examples

- [Examples](#shallow)
    - [Shallow](#shallow)
    - [Utils](#utils)
    - [Warnings](#warnings)

## Shallow

### Feedforward Neural Network (F_NN)

#### 1. Create a Feedforward Neural Network

You can create a simple feedforward neural network like this:

```python
from template_nn.shallow.fnn import F_NN

model = F_NN(
    input_size=12,
    output_size=4,
    hidden_sizes=[10, 7]
)
```

#### 2. Define Model Using a Configuration Dictionary

Alternatively, define the configuration as a dictionary:

```python
from template_nn.shallow.fnn import F_NN

config = {
    "input_size": 12,
    "output_size": 4,
    "hidden_sizes": [10, 7]
}

model = F_NN(tabular=config)
```

#### 3. Define Model Using a DataFrame

You can also use a DataFrame for the configuration:

```python
import pandas as pd
from template_nn.shallow.fnn import F_NN

config = pd.DataFrame({
    "input_size": [12],
    "output_size": [4],
    "hidden_sizes": [[10, 7]]  # List as a value
})

model = F_NN(tabular=config)
```

#### 4. Use Custom Activation Functions

The default activation function is `nn.ReLU()`, but you can pass your own, such as `nn.Tanh()`:

```python
import torch.nn as nn
from template_nn.shallow.fnn import F_NN

model = F_NN(
    input_size=12,
    output_size=4,
    hidden_sizes=[10, 7],

    # number of activation functions must be
    # equal to how many hidden layers you have
    activation_functions=[nn.Tanh(), nn.ReLU()]
)
```

#### 5. Visualize the Model Architecture

Optionally, you can visualize the model layers:

```python
from template_nn.shallow.fnn import F_NN

model = F_NN(
    input_size=12,
    output_size=4,
    hidden_sizes=[10, 7],
    visualise=True
)
```

This will output the model architecture:

```console
F_NN(
  (model): Sequential(
    (0): Linear(in_features=12, out_features=10, bias=True)
    (1): ReLU()
    (2): Linear(in_features=10, out_features=7, bias=True)
    (3): ReLU()
    (4): Linear(in_features=7, out_features=4, bias=True)
  )
)
```

## Utils

### get_gpu_device

This utility function automatically detects and uses your GPU for training:

```python
from template_nn.utils.gpu import get_gpu_device

gpu = get_gpu_device()
```

## Warnings

By default, the model warns you about different model architectures:

```console
UserWarning: *** Shallow Neural Network Detected ***
A shallow neural network (<= 2 hidden layers) is being used.

If you need more complexity or better performance, consider
switching to a deeper architecture (>= 3 hidden layers).
```

If you prefer to suppress all warnings, add this at the top of your file or notebook:

```python
import warnings

warnings.filterwarnings("ignore")
```

See [this](https://stackoverflow.com/questions/14463277/how-to-disable-python-warnings) on StackOverflow for more
information on suppressing individual warnings.
