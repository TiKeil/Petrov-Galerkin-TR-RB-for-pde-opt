```
# ~~~
# This file is part of the paper:
#
#   "Model Reduction for Large Scale Systems"
#
#   https://github.com/TiKeil/Petrov-Galerkin-TR-RB-for-pde-opt
#
# Copyright 2019-2021 all developers. All rights reserved.
# License: Licensed as BSD 2-Clause License (http://opensource.org/licenses/BSD-2-Clause)
# Authors:
#   Tim Keil (2019 - 2021)
# ~~~
```

In this repository, we provide jupyter-notebooks and the entire code for the numerical experiments in Section 3 of the paper 
"Model Reduction for Large Scale Systems"
by Tim Keil and Mario Ohlberger. 

For just taking a look at the provided (precompiled) jupyter-notebooks, you do not need to install the software.
Just go to [`notebooks/Lssc21_simulations`](https://github.com/TiKeil/Petrov-Galerkin-TR-RB-for-pde-opt/tree/master/notebooks).
If you want to have a closer look at the implementation or compile the results by
yourself, we provide simple setup instructions for configuring your own Python environment in a few steps.
We note that our setup instructions are written for Linux or Mac OS only and we do not provide setup instructions for Windows.
We also emphasize that our experiments have been computed on a fresh Ubuntu 20 system with Python version 3.8.5. with 12 GB RAM. 

# Organization of the repository

Our implementation is based on pyMOR (https://github.com/pymor/pymor).
Further extensions that we used for this paper can be found in the directory [`pdeopt/`](https://github.com/TiKeil/Petrov-Galerkin-TR-RB-for-pde-opt/tree/master/pdeopt). 

# How to find figures and tables from the paper

We provide instructions on how to find all figures and tables from the paper. 

**Figure 1**: The data of the blueprint is in [`EXC_data/`](https://github.com/TiKeil/Petrov-Galerkin-TR-RB-for-pde-opt/tree/master/EXC_data). 
The used file for Figure 1 is `full_diffusion_with_big_numbers_with_D.png`

**Figure 2**: The result is a collection of data from 
[`here`](https://github.com/TiKeil/Petrov-Galerkin-TR-RB-for-pde-opt/blob/master/notebooks/Lssc21_simulations/estimator_study/) where all trainings are performed and shown. 

**Figure 3**: This result is based on starting value seed 9 (Starter9)
[`here`](https://github.com/TiKeil/Petrov-Galerkin-TR-RB-for-pde-opt/blob/master/notebooks/Lssc21_simulations/optimization_results/).
The figure and the table can be constructed by the scripts in the corresponding `results/` directory.

# Setup

On a Linux or Mac OS system with Python and git installed, clone
the repo in your favorite directory

```
git clone https://github.com/TiKeil/Petrov-Galerkin-TR-RB-for-pde-opt
```

Just run the provided setup file via 

```
cd Petrov-Galerkin-TR-RB-for-pde-opt
./setup.sh
```

# Running the jupyter-notebooks

If you want to interactively view or compile the notebooks, just activate and start jupyter-notebook 

```
source venv/bin/activate
jupyter-notebook --notebook-dir=notebooks
```

We recommend to use notebook extensions for a better overview in the notebooks.
After starting the jupyter-notebook server go to Nbextensions, deactivate the first box and activate at least `codefolding` and `Collapsible Headings`. 
