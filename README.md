<div align="center">
  <h1>Template NN</h1>
  <p><strong>A lightweight, declarative, and opinionated neural network library for PyTorch</strong></p>

  [![GitHub release (latest by date)](https://img.shields.io/github/v/release/gabrielchoong/template-nn)](https://github.com/gabrielchoong/template-nn/releases)
  [![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
</div>

<hr/>

**Template NN** is a lightweight, easy-to-use library designed to streamline the learning process and implementation of machine learning models. It provides an opinionated, declarative interface for rapidly scaffolding architectures like Convolutional Neural Networks (CNNs) and Feedforward Neural Networks (FNNs) while maintaining **100% full compatibility** with existing PyTorch code. 

If you love the flexibility of PyTorch but want to reduce boilerplate when prototyping, this library is for you.

Huge thanks to the [PyTorch](https://github.com/pytorch/pytorch) team for enabling projects like this!

---

## Key Features

* **Declarative Architectures**: Scaffold complex networks in a single line. 
* **Zero Magic**: Acts as a transparent, drop-in wrapper over native `torch.nn.Module`. You still have full access to the underlying PyTorch abstractions.
* **Readable & Expressive**: Uses intuitive acronyms (e.g., `CNN`, `FNN`) and straightforward arguments to let you focus on ML concepts rather than plumbing.
* **Prototyping & Benchmarking**: Quickly test hypotheses and benchmark model variants with minimal code churn.

## Installation

> [!NOTE]
> The PyPI package for `template-nn` is deprecated (frozen at 0.2.3). Moving forward, all new versions (like `0.3.0`) will be published exclusively as GitHub releases. 

### Directly from GitHub (Recommended)

Install the latest stable release directly from this repository:

```sh
pip install git+https://github.com/gabrielchoong/template-nn.git@v0.3.0
```
*(To install the cutting-edge `main` branch, omit the `@v0.3.0` tag).*

### From Source (For Development)

If you wish to modify the library or contribute, clone the repository and install it locally. We recommend using [`uv`](https://github.com/astral-sh/uv) for fast and reliable environment management.

```sh
git clone https://github.com/gabrielchoong/template-nn.git
cd template-nn

# Using uv (Recommended)
uv venv
uv sync
```

*Alternative using `pip` (not recommended for active contributing):*
```sh
pip install -r requirements.txt
pip install -e .
```

## Quick Start & Usage

*More detailed examples coming soon.* 

The library uses shortened acronyms for common architectures. For example, you can instantiate a Convolutional Neural Network declaratively:

```python
import torch
from template_nn import CNN

# Example declarative single-line instantiation 
# (Check source docs for exact signature based on version)
model = CNN(
    {} # your configs in here
)

# Since it inherits from torch.nn.Module, use it just like a native PyTorch model!
dummy_input = torch.randn(1, 3, 224, 224)
output = model(dummy_input)
print(output.shape)
```

Because **Template NN** avoids proprietary "magic", the returned `model` is a fully compliant PyTorch Module. You can integrate it directly with your existing PyTorch training loops, dataloaders, and optimizers without any friction.

## Contributing

We welcome contributions! As this project is actively evolving, please expect breaking changes between minor versions.

1. See our [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines on how to get started.
2. Check the [CHANGELOG.md](CHANGELOG.md) to track recent updates and breaking changes.

##  License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
