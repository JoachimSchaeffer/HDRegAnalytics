Author: Joachim Schaeffer
Email: joachim.schaeffer@posteo.de

ReadMe for repository acssocaited with the article: 
Interpretation of High-Dimensional Linear Regression — Effects of Nullsapce and Regularization Demonstrated on Battery Data

## Code

All plots associated with the paper can be generated by the notebooks. 
The code for objects, methods and functions is in the src folder.

Example on fully synthetic parabolic data:
- [Nullspace_Parabola_Examples.ipynb](Nullspace_Parabola_Examples.ipynb)

Example on Lithium-Iron-Phosphate Cycling data:
- [Nullspace_LFP_examples.ipynb](Nullspace_LFP_examples.ipynb)

## LFP Data Files

The measurement data is contained in the file [lfp_slim.csv](lfp_slim.csv) and the corresposning license for this data is [lfp_datalicense.txt](lfp_datalicense.txt).

## Environments

All this python code and tooling has dependencies which are encoded in the environment files. 
To install the anaconda environment, you need to have anaconda installed, then run:
```shell
conda env create --file python_environments/environment.yml
conda activate HDRegAnalytics
```
Known issue: You need to have a working installation of latex and all other requirtements for matplotlib wokring with latex. 
More information here [MatplotlibLatex](https://matplotlib.org/stable/tutorials/text/usetex.html).

We recommend using R-Studio for running the R code contained in the folder regression_in_R. 
