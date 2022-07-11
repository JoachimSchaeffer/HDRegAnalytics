Author: Joachim Schaeffer
ReadMe for repository accompanying [this publication]

## Code

Examples of using this code can be found in [notebooks/](notebooks/), and also in the automated tests. The tests may be helpful for understanding the interfaces and behaviors of the functions. In particular, [eis/test_circuit_parsing.py](eis/test_circuit_parsing.py) contains working examples of the flexibility in providing parameters to circuit models, and how to register a new circuit model if you'd like to do so. There are utility functions for reading and writing the EIS data csvs, which we recommend you use to generate model output for your submission in [EISDataIO.py](eis/EISDataIO.py). There are also utility functions for plotting EIS spectra, with and without ECM, in [EISPlot.py](eis/EISPlot.py). If you'd like to investigate the ECM source code, and perhaps register a new circuit, see [EquivalentCircuitModels.py](eis/EquivalentCircuitModels.py).

Example notebooks:

- [eis_plot_and_ecm_fit_examples.ipynb](notebooks/eis_plot_and_ecm_fit_examples.ipynb).
  - Examples of inspecting the EIS data, creating ECMs from it
- [eis_ecm_guess_scoring_correct_ECM.ipynb](notebooks/eis_ecm_guess_scoring_correct_ECM.ipynb).
  - Examples of scoring parameter guesses and fitting parameters when you've got the correct ECM
- [eis_ecm_guess_scoring_wrong_ECM.ipynb](notebooks/eis_ecm_guess_scoring_wrong_ECM.ipynb).
  - Examples of scoring parameter guesses and fitting parameters when you've got the wrong ECM

## LFP Data Files

The EIS data for training and validating your model(s) has been provided to you in [train_data.csv](train_data.csv). This data file contains rows of EIS data with Frequencies, Impedances, QuantumScape identified ECM classes, and QuantumScape optimized ECM parameters. This consists of 7462 rows of already-labeled EIS data provided by QuantumScape.

We also have a dataset of never-labeled EIS data, [unlabeled_data.csv](unlabeled_data.csv). This data file contains 18569 rows of unlabeled EIS spectra provided by QuantumScape.

An example of the format we're expecting for the challenge submission is provided at [submission_example.csv](submission_example.csv), and the [Challenge Description](Challenge.md) has more wordy detail on the format. It's generated as simple copy from the Circuit and Parameters columns of training data file.

## Environments

All this python code and tooling has dependencies which are encoded in the environment files. We've set up support for [Anaconda](https://anaconda.org/), [Poetry](https://python-poetry.org/), and [Docker](https://www.docker.com/) environments. Even if you're not primarily using python for your development, we encourage you to setup a python environment just to use DVC and our scoring ClI (see next section). If you are using another programming language, please provide details on your environment in one of the commonly accepted formats, and a brief description of how to set it up.

To install the anaconda environment, you need to have anaconda installed, then run:

```shell
conda env create --file environments/environment.yml
```

To install the poetry environment, you need to have python 3.10 or higher installed, then run:

```shell
cp environments/pyproject.toml .; poetry install
```

To install the Docker environment, you need to have Docker installed, then run:

```shell
cp environments/docker-compose.yml .
cp environments/Dockerfile .
cp environments/juypter.sh .
docker-compose build
```

The Docker container is setup to run a juypter lab server on start. You can start it with `docker-compose up`.

After you picked and setup an environment in your repository, please delete the other environment files so the judges know which one you used when judging the submissions.
