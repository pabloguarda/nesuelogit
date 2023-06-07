Neural Estimation of Stochastic User Equilibrium with LOGIT assignment models (nesuelogit)
==============================================================================

Development Setup
=================

1. Clone this repository.

2. Download and install Anaconda for 64-Bit (M1): https://docs.anaconda.com/anaconda/install/index.html
3. Create virtual environment: ``conda create -n nesuelogit``
4. Activate environment: ``conda activate nesuelogit``
5. Install dependencies: ``conda env update -f nesuelogit-cpu.yml`` or ``conda env update -f nesuelogit-gpu.yml`` to
train models with gpu.

This repository is currently compatible with Python 3.8.x and it requires a Macbook with Apple Silicon chip.


# To enable gpu training: conda env update -f nesuelogit-gpu.yml --prune
