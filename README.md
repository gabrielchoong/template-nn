# Template NN

Template NN is a lightweight, easy-to-use template designed to serve as a drop-in replacement for PyTorch's core neural network modules, streamlining prototyping and development. It aims to provide a more opinionated and simplified interface while fully leveraging PyTorch's powerful backend.

Huge thanks to the PyTorch team for enabling projects like this.


## Installation

> [!WARNING]
> **Breaking Changes in Version>=0.1.6**
>
> This release includes the removal of optimiser support, directory renames, module restructuring, and function removals.
> If you're upgrading from an earlier version, please read the [Release Notes](RELEASE.md) before updating.

You can install `template-nn` via pip:

```sh
pip install template-nn
```

Or clone the repository and install it locally:

```sh
git clone https://github.com/gabrielchoong/template-nn.git
cd template-nn
pip install -r requirements.txt
pip install .
```

## Documentation

For detailed documentation, including usage instructions and examples, visit the online documentation
at [Documentation](https://gabrielchoong.github.io/template-nn).

## Examples

For runnable examples demonstrating how to use `template-nn`, please refer to the `examples/` directory in this repository.

### Feature Preview

```python
import torch
import torch.nn as nn
from template_nn import CNN

# --- Using raw PyTorch (Simple CNN) ---
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(nn.functional.relu(self.conv1(x)))
        x = self.pool(nn.functional.relu(self.conv2(x)))
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = nn.functional.relu(self.fc1(x))
        x = nn.functional.relu(self.fc2(x))
        x = self.fc3(x)
        return x

pytorch_cnn_model = SimpleCNN()

# --- Using Template NN (Simple CNN) ---
template_nn_cnn_model = CNN({
    "image_size": (32, 32), # Example input image size
    "conv_channels": [3, 6, 16],
    "conv_kernel_size": 5,
    "pool_kernel_size": 2,
    "fcn_hidden_sizes": [120, 84],
    "activation_functions": [nn.ReLU(), nn.ReLU(), nn.ReLU(), nn.ReLU()], # For conv and fc layers
    "output_channel": 10
})

print("PyTorch CNN Model Architecture:")
print(pytorch_cnn_model)
print("\nTemplate NN CNN Model Architecture:")
print(template_nn_cnn_model)
```

## Releases and Contributing

**This project is currently in its beta stage**.
Please [file an issue](https://github.com/gabrielchoong/template-nn/issues) if you found a bug.

To read about the motive and direction of this project, see [Roadmap](ROADMAP.md). To read more about the current
releases, see [Release Notes](RELEASE.md).

Contributions are welcomed! Please see [Contributions](CONTRIBUTING.md) for information on contributing.

### Contributors

This project is currently being developed by [Gabriel](https://github.com/gabrielchoong).

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
