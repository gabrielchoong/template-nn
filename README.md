# Template NN

Template NN is a lightweight, easy-to-use library designed to streamline the learning process of machine learning. It aims to provide a more opinionated interface while maintaining full compatibility with any existing PyTorch code.

Huge thanks to the [PyTorch](https://github.com/pytorch/pytorch) team for enabling projects like this.

## Purpose

Initially developed for my thesis to simplify the codebase, but evolved to a library with a declarative interface for testing and benchmarking neural networks.

The classes uses shortened acronyms such as `CNN` for *Convolution Neural Network* with single line instantiation to better focus on the learning aspect of ML.

## Installation

Install this library from PyPI:

```sh
pip install template-nn
```

Clone the repository and install it locally for development:

```sh
git clone https://github.com/gabrielchoong/template-nn.git
cd template-nn
```

Using `uv` (recommended):

```sh
uv venv
uv sync
```

Using `pip` (not recommended for contributing):

```sh
pip install -r requirements.txt
pip install .
```

## Releases and Contributing

See [changelog](CHANGELOG.md) for previous changes. Expect breaking changes at this stage.

See [contributing](CONTRIBUTING.md) if you wish to contribute to this project.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
