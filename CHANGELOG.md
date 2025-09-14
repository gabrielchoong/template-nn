# Release Notes

## Stable

### 0.2.0

> This version will introduce changes that would otherwise be incompatible with v0.1.x of this library. If you depend on v0.1.x, consider locking the version for this library to the latest v0.1.x branch, which is `0.1.6`.

**Breaking Changes**:

- **33e059f**: Removed `pandas.Dataframe` support in model configuration to better streamline the process for maintaining the codebase. The `pandas` dependency has also been dropped from the project entirely.
- **229e5f4**: `FNN` - Changed the type of `model_config["activation_functions"]` from `list[nn.Module]` to `list[str]` to simplify constructor interface.

**Improvements**:

- **c536c6e**: `forge.py` has been updated to the most recent class structure syntax.
- **c536c6e**: `forge.py` now includes a hint in generated tests files to prompt the addition of `<classname>` into `src/__init__.py`.
- **99c6131**: Typing variables now use Python's native type annotations.

**Changes**:

- **dbd6b8e**: Renamed `RELEASE.md` to `CHANGELOG.md` to better communicate changes made.
- **a0748f4**: Added `ruff` as the formatter and linter of choice for this project.
- **fcee959**: Added `uv` as the dependency and package manager in favour of (ana)conda.
- **44508c6**: Updated the licensing year in `LICENSE`.
- **eb02f32**: Removed `examples` directory from `src/` due to changing class interfaces.
- **82dcdc1**: Removed `mealpy` as project dependency.
- **37c5f31**: Removed `ROADMAP.md` as it no longer aligns with project direction.
- **33e059f**: Removed `pandas` as project dependency.
- **c536c6e**: Refactored `BaseNetwork` and `forge`.
  - `BaseNetwork` is now an abstract class with abstract methods inherited from the `abc.ABC` class.
  - `BaseNetwork.__init__` no longer implicitly invokes `self._build_model()`.
  - Updated docstrings to be more meaningful for contributors.
  - `forge` now works correctly to create network and test stubs.
    - Created tests provides hints to add `<classname>` into `src/__init__.py`.
- **229e5f4**: 
  - Refactored all neural network classes to use consistent styling and code organisation.
  - Added an implementation of `getattr(torch.nn, activation_function)()` instead of passing raw nn.Module functions.
- **99c6131**: Refactored `typing.List` to `list` in type annotations.
- **790ef14**: Updated return type of `get_model_keys` from `list` to `tuple[str]`.
- **b3b6ac3**: Refactored outdated tests.
- **22d6ebd**: Moved the handling of visualisation of created neural networks out of the `BaseNetwork` class.
- **c88054d**: Classes inheriting `BaseNetwork` now implements the visualisation separately.
- **0a280d7**: `forge` now includes the visualisation step when creating new classes using `--install`.

## Stable

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