Author: Joachim Schaeffer
Email: joachim.schaeffer@posteo.de

ReadMe for repository accompanying [this publication]

## Code

Examples of using the described methodolgy code can be found in [notebooks/](notebooks/). There are a couple of different examples build in and demonstrated. 
Synthetically generated data on the one hand, as well as measurement data with cosntructed ground truth values. 

Example notebooks:

- [nullspace_methodology_example.ipynb](nullspace_methodology_example.ipynb).

## LFP Data Files

The measurement data is contained in the file [lfp_slim.csv](lfp_slim.csv) and the corresposning license for this data is [lfp_datalicense.txt](lfp_datalicense.txt).

## Environments

All this python code and tooling has dependencies which are encoded in the environment files. We've set up support for [Anaconda](https://anaconda.org/), and [Docker](https://www.docker.com/) environments. 

To install the anaconda environment, you need to have anaconda installed, then run:
```shell
conda env create --file environments/environment.yml
```
Known issue: You need to have a working installation of latex and all other requirtements for matplotlib wokring with latex. 
More information here [MatplotlibLatex](https://matplotlib.org/stable/tutorials/text/usetex.html).

To install the Docker environment, you need to have Docker installed, then run:

```shell
cp environments/docker-compose.yml .
cp environments/Dockerfile .
cp environments/jupyter.sh .
cp environments/requirements.txt .
docker-compose build
```

The Docker container is setup to run a juypter lab server on start. You can start it with `docker-compose up`.

Known issues: The docker containre is relatively large due to the latex installation. Furthermore, it might take a while for the plots to render when using docker (depending on the resources you assign to the container). 