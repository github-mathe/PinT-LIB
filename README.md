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
pybamm_lithium/
├── Parameter/
│   └── Javid/          # Cell parameters for both half and full cells
└── testcases/          # Test case implementations (this directory)
    │----cccv_half_cell.ipynb
    │----gitt_half_cell.ipynb
    │----cccv_full_cell.ipynb
    │----gitt_full_cell.ipynb
```

## Prerequisites

### Required Software

1. **PyBaMM** - Battery modeling framework
2. **Jupyter Notebook** - For running the test case notebooks

For detailed installation instructions, visit the [PyBaMM Installation Guide](https://docs.pybamm.org/en/latest/source/user_guide/installation/index.html).

## Cell Parameters

The cell parameters used in these test cases are based on the research published in:

**DOI:** [10.1016/j.electacta.2024.144259](https://doi.org/10.1016/j.electacta.2024.144259)

## Solver Configuration

PyBaMM allows you to change the numerical solver used for simulations. You can modify the solver in the notebooks by specifying different solver options when calling the `solve()` method.

## Learn More

### PyBaMM Documentation
- [PyBaMM Documentation](https://docs.pybamm.org/)
- [PyBaMM GitHub Repository](https://github.com/pybamm-team/PyBaMM)

### Battery Testing Protocols
- [CCCV Charging Protocol](https://docs.pybamm.org/en/v23.5_a/source/examples/notebooks/simulating-long-experiments.html)
- [GITT Technique](https://doi.org/10.1002/adts.202500302)
- [Lithium-Ion Battery Modeling with PyBaMM](https://docs.pybamm.org/en/latest/source/examples/index.html)

