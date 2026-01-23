# PyBaMM Lithium Test Cases

This directory contains test cases for lithium-ion battery modeling using PyBaMM (Python Battery Mathematical Modelling).

## Overview

The test cases simulate two types of cells:
- **Half Cell**: Contains electrochemical testing scenarios for half-cell configurations
- **Full Cell**: Contains electrochemical testing scenarios for full-cell configurations

For each cell type, the following test protocols are implemented:
- **CCCV (Constant Current Constant Voltage)**: Standard charging protocol
- **GITT (Galvanostatic Intermittent Titration Technique)**: Used for measuring diffusion coefficients and thermodynamic properties

## Directory Structure

```
batpint/            # Python library for parameters and base code
└── parameter/
    └── javid/
        │---Li_full.py
        │---Li_half.py
scripts/            # Test case implementations
└── notebooks/
    │---cccv_half_cell.ipynb
    │---gitt_half_cell.ipynb
    │---cccv_full_cell.ipynb
    │---gitt_full_cell.ipynb
└── notebooks
    │---01_baseCase.py  # script to simulate every cases in the notebooks
```

## Installation

All test cases implemented in the [`scripts`](./scripts) folder relies on code implemented in the `batpint` python package.
It requires `python>=3.10` and can be simply installed on your own python environment using :

```bash
pip install --no-deps -e .
```

> 💡 The `-e` options install the `batpint` package in editable mode : any modification on the code will be automatically taken into account when updating the package import.

Note that the previous command install without the packages dependencies (like `pybamm`, `numpy`, `matplotlib`, see [pyproject.toml](./pyproject.toml) for the complete list). 
To install all dependencies at the same time (or check that they are installed), just run :

```bash
pip install -e .
```

### Required Software

1. **PyBaMM** - Battery modeling framework
2. **Jupyter Notebook** - For running the test case notebooks

For detailed installation instructions, visit the [PyBaMM Installation Guide](https://docs.pybamm.org/en/latest/source/user_guide/installation/index.html).

## Cell Parameters

The cell parameters used in these test cases are based on the research published in:

**DOI:** [10.1016/j.electacta.2024.144259](https://doi.org/10.1016/j.electacta.2024.144259)

## Discretization and Spatial Methods

PyBaMM uses finite volume methods as the default spatial discretization technique for solving the governing equations. This method is well-suited for battery modeling as it conserves fluxes across domain boundaries.

### Changing the Spatial Discretization Method

The spatial discretization method can be customized for different domains if needed. The default spatial methods are:

- **macroscale**: Finite Volume
- **negative particle**: Finite Volume
- **positive particle**: Finite Volume
- **current collector**: Zero-Dimensional Spatial Method

To change the spatial discretization method, you can modify the `default_spatial_methods` dictionary before discretizing the model. For example:

```python
# Get the default spatial methods
spatial_methods = model.default_spatial_methods

# Change the discretization method for a specific domain
spatial_methods["negative particle"] = pybamm.SpectralVolume()

# Use the modified spatial methods when discretizing
disc = pybamm.Discretisation(mesh, spatial_methods)
```

For detailed examples and more information on customizing discretization settings, see the [PyBaMM documentation on changing settings](https://docs.pybamm.org/en/stable/source/examples/notebooks/change-settings.html#Changing-the-discretisation).

## Solver Configuration

PyBaMM allows you to change the numerical solver used for simulations. You can modify the solver in the notebooks by specifying different solver options when calling the `solve()` method.

For better performance and accuracy, it is recommended to use the **IDAKLU solver** instead of the default solver. IDAKLU is based on SUNDIALS and provides superior numerical stability and convergence properties for battery modeling problems.

```python
# Use IDAKLU solver for improved performance
solver = pybamm.IDAKLUSolver()
solution = solver.solve()
```

For more details on available solvers and their characteristics, refer to the [PyBaMM solver documentation](https://docs.pybamm.org/en/latest/source/api/solvers/index.html).

## Learn More

### PyBaMM Documentation
- [PyBaMM Documentation](https://docs.pybamm.org/)
- [PyBaMM GitHub Repository](https://github.com/pybamm-team/PyBaMM)

### Battery Testing Protocols
- [CCCV Charging Protocol](https://docs.pybamm.org/en/v23.5_a/source/examples/notebooks/simulating-long-experiments.html)
- [GITT Technique](https://doi.org/10.1002/adts.202500302)
- [Lithium-Ion Battery Modeling with PyBaMM](https://docs.pybamm.org/en/latest/source/examples/index.html)

