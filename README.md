# Template NN
Template NN is a lightweight, easy-to-use template built to streamline neural network prototyping. At its core, it leverages the power of [PyTorch](https://github.com/pytorch/pytorch).

Huge thanks to the PyTorch team for enabling projects like this.

## Installation

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

For detailed documentation, including usage instructions and examples, visit the online documentation at [Documentation](https://gabrielchoong.github.io/template-nn).

### Feature Preview

For more information on how to use this project, see [Examples](EXAMPLES.md).

```python
from template_nn.shallow.fnn import F_NN

model = F_NN(input_size=10, output_size=5, hidden_sizes=5)
# model = F_NN(input_size=10, output_size=5, hidden_sizes=[8, 6])
```

## Releases and Contributing

**This project is currently in its alpha stage**. Please [file an issue](https://github.com/gabrielchoong/template-nn/issues) if you found a bug.

To read about the motive and direction of this project, see [Roadmap](ROADMAP.md). To read more about the current releases, see [Release Notes](RELEASE.md).

Contributions are welcomed! Please see [Contributions](CONTRIBUTING.md) for information on contributing.

### Contributors

This project is currently being developed by [Gabriel](https://github.com/gabrielchoong) and [Benjamin](https://github.com/Ben1001409).

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
