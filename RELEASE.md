# Releases Notes

## Alpha

### 0.1.4

**Breaking Changes (dev only)**:
- Moved `utils.gpu.gpu.get_gpu_device` to `utils.gpu.get_gpu_device`.
- Added a new `keys` argument in `utils.model_compose.build_tabular_model` (required if you're calling this directly or have overriden internals).

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

- Added `shallow.fnn`: Easily create customisable feedforward neural networks with support for dynamic hidden layer configurations.
- Added `utils.gpu`: Introduced utility functions for seamless GPU setup and integration, enabling faster computation and streamlined workflows.