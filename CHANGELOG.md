# Release Notes



### 0.1.6

**Breaking Changes**:

- Removed optimiser support: The optimiser functionality has been deprecated and removed.

**New Features**:

- Introduced a new neural network template.
- Model keys are now stored in a `keys.json` file instead of a Python dictionary.

**Improvements**:

- Updated `README.md` for better clarity and information.
- Refactored imports for improved code organization.
- Moved files from the `networks` directory for better structure.
- Added additional unit tests to further enhance code reliability.

### 0.1.5

- Improved build system: Added `__init__.py` to ensure `_utils` is recognized during build processes.
- Enhanced code quality:
  - Added new unit tests to improve test coverage and stability.
  - Applied comprehensive Python linting and formatting across the codebase.

### 0.1.4

**Breaking Changes**:

- Removed `utils.gpu.gpu.get_gpu_device`.
- Removed `utils.model_compose_utils`.
- Renamed directories: `utils/` ➝ `_utils/`, and `shallow/` ➝ `networks/`.
- Removed unnecessary abstractions, including `_utils.model_compose` and `_utils.model_compose_utils`.
- Added `FNN` to the public API via `__init__.py`.
  - Users can now import directly using `from template_nn import FNN`.
- Introduced `BaseNetwork`, a new base class designed to support all future neural network architectures.
- Improved error handling:
  - Explicit error messages are now raised instead of being silently handled.

### 0.1.3

- Added `optimisers.hho`: Optimise an objective function based on the Harris' Hawks Optimiser algorithm.
- Added `utils.solution`, `utils.hho_operations` and `utils.levy`: Helper function for the `HHO` class.
- Changed how dictionary keys are handled in `utils.model_compose_utils.get_params`.
- Temporarily added keys tuple in `utils.model_compose.build_tabular_model`.

### 0.1.2

- Updated `utils.args_val.validate_args` to perform additional checks.
- Added a `visualise` option in `shallow.fnn.F_NN`.
- Added code documentation in all files.
- Split the codebase into a more maintainable structure.
- Added documentation: `README`, `ROADMAP`, `RELEASE`, `CONTRIBUTING` and `EXAMPLES`.

### 0.1.1

- Added ability to modify activation functions in `shallow.fnn.F_NN`.
- Added ability to create a model using a `dict` or a `pandas.DataFrame` in `shallow.fnn.F_NN`.

### 0.1.0

- Added `shallow.fnn`: Easily create customisable feedforward neural networks with support for dynamic hidden layer
  configurations.
- Added `utils.gpu`: Introduced utility functions for seamless GPU setup and integration, enabling faster computation
  and streamlined workflows.