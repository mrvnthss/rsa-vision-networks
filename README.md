# rsa-vision-networks

[![Python](https://img.shields.io/badge/Python-3.11-3776AB.svg?style=flat&logo=python&logoColor=white)](https://www.python.org)
[![Poetry](https://img.shields.io/endpoint?url=https://python-poetry.org/badge/v0.json)](https://python-poetry.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch_2.4.1-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/get-started/locally/)
[![GitHub License](https://img.shields.io/github/license/mrvnthss/brightness-discrimination-2afc?color=9a2333)](https://opensource.org/license/mit/)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.14693502.svg)](https://doi.org/10.5281/zenodo.14693502)

> This repository contains the source code for my master's thesis on "**A Quantitative Evaluation of Representational Similarity Analysis in Image Classification Networks**", submitted in the Department of General Psychology at the [University of Giessen](https://www.uni-giessen.de/en) in partial fulfillment of the requirements for the degree of *Master of Science* in "[*Mind, Brain and Behavior*](https://www.uni-giessen.de/de/studium/studienangebot/master/mbb)".

<div align="center">
    <img src="https://github.com/mrvnthss/rsa-vision-networks/blob/main/reports/figures/lenet_fashionmnist/representational_similarity/analysis/lenet_geometry.png?raw=true" alt="card-game-dobble" width="700">
</div>

## Table of Contents

- [Getting Started](#getting-started)
- [Project Structure](#project-structure)
- [Coding Style](#coding-style)
- [License](#license)

## Getting Started

This section will guide you through setting up the project locally.

### Prerequisites

- Python 3.11 or higher
- Git (optional, for cloning the repository)

### Installing Poetry

This project uses Poetry for dependency management. If you don't have Poetry installed, you can install it using one of the following methods:

**Linux, macOS, Windows (WSL)**:
```
curl -sSL https://install.python-poetry.org | python3 -
```

**Windows (PowerShell)**:
```
(Invoke-WebRequest -Uri https://install.python-poetry.org -UseBasicParsing).Content | py -
```

**Alternative methods**:
- Using `pip` &rarr; `pip install poetry`
- Using Homebrew (macOS) &rarr; `brew install poetry`
- Using Scoop (Windows) &rarr; `scoop install poetry`

Verify your installation:
```
poetry --version
```

For detailed installation instructions and troubleshooting, visit the [Poetry documentation](https://python-poetry.org/docs/#installation).

### Installation

1. Clone the repository (or download and extract the ZIP file):
   ```
   git clone https://github.com/mrvnthss/rsa-vision-networks.git
   cd rsa-vision-networks
   ```

2. Install dependencies using Poetry:
   ```
   poetry install
   ```

   This will create a virtual environment and install all required dependencies specified in `pyproject.toml`.

3. Activate the virtual environment:
   ```
   poetry shell
   ```

## Project Structure

```
rsa-vision-networks/
├── data
│   ├── processed
│   └── raw
├── models
├── notebooks
│   ├── 0.2-mt-misc.ipynb
│   ├── 1.1-mt-basic-training-routine.ipynb
│   ├── 2.0-mt-parsing-tensorboard-data.ipynb
│   ├── 3.0-mt-pre-trained-weights.ipynb
│   ├── 4.12-mt-data-analysis-lenet-fashionmnist.ipynb
│   ├── 5.4-mt-rsa-lenet-fashionmnist.ipynb
│   ├── 6.3-mt-differentiable-spearman.ipynb
│   ├── 7.1-mt-rsa-modified-lenet-fashionmnist.ipynb
│   ├── 8.0-mt-optuna-studies.ipynb
│   ├── 9.2-mt-analyzing-results-main-experiments.ipynb
│   └── hist
├── out
├── reports
│   ├── data
│   └── figures
├── src
│   ├── base_classes
│   ├── conf
│   │   ├── criterion
│   │   ├── dataset
│   │   ├── experiment
│   │   ├── main_scheduler
│   │   ├── model
│   │   ├── optimizer
│   │   ├── optuna
│   │   ├── performance
│   │   ├── repr_similarity
│   │   ├── transform
│   │   ├── warmup_scheduler
│   │   ├── compute_dataset_stats.yaml
│   │   ├── test_classifier.yaml
│   │   ├── train_classifier.yaml
│   │   ├── train_classifier_cv.yaml
│   │   ├── train_classifier_optuna.yaml
│   │   ├── train_similarity.yaml
│   │   └── train_similarity_cv.yaml
│   ├── dataloaders
│   ├── datasets
│   ├── models
│   ├── rsa
│   ├── schedulers
│   ├── training
│   ├── utils
│   ├── visualization
│   ├── __init__.py
│   ├── compute_dataset_stats.py
│   ├── config.py
│   ├── test_classifier.py
│   ├── train_classifier.py
│   ├── train_classifier_cv.py
│   ├── train_classifier_optuna.py
│   ├── train_similarity.py
│   └── train_similarity_cv.py
├── LICENSE
├── optuna_studies.sqlite3
├── poetry.lock
├── pyproject.toml
└── README.md
```

- `data/` &rarr; Image datasets used in the project.
- `models/` &rarr; Model checkpoints along with corresponding training configurations and log files.
- `notebooks/` &rarr; Jupyter notebooks for data exploration, analysis, and visualization.
- `out/` &rarr; Output files (logs, checkpoints) generated when running experiments.
- `reports/` &rarr; Figures created for the thesis, raw data from simulations.
- `src/` &rarr; Source code for the project.
- `LICENSE` &rarr; MIT License file.
- `optuna_studies.sqlite3` &rarr; SQLite database containing the results of hyperparameter optimization studies.
- `poetry.lock` &rarr; Dependency lock file automatically created by Poetry.
- `pyproject.toml` &rarr; Project configuration file for dependency management with Poetry.

> [!IMPORTANT]
> The directories `data/`, `out/`, and `report/data` as well as the `optuna_studies.sqlite3` database are not included in this repository. Raw image data are automatically downloaded and placed in the `data/` directory when datasets are first initialized. Output files such as model checkpoints, log files, TensorBoard logs, and simulated data, as well as the SQLite database `optuna_studies` are available for download from [Zenodo](https://doi.org/10.5281/zenodo.14693502).

## Coding Style

- [PEP8](https://peps.python.org/pep-0008/) formatting
- Generally following the [Google Python Style Guide](https://google.github.io/styleguide/pyguide.html)
- Maximum line length of 99 characters (72 characters for docstrings) in source code
- Maximum line length of 119 characters in Jupyter notebooks

## License

This project is licensed under the **MIT License**. The MIT License is a permissive free software license that offers significant freedom to use, modify, distribute, and sell the software and its derivatives, with minimal restrictions. Key points of the MIT License include:

- **Permission to Use**: You are free to use the software for any purpose, including commercial applications.

- **Permission to Modify**: You have the right to modify the software in any way you see fit.

- **Permission to Distribute**: You can distribute the software and your modifications to anyone, under the same license terms.

- **Permission to Sublicense**: You can grant a sublicense to modify and distribute the software to others under your own terms.

- **No Warranty**: The software is provided "as is", without warranty of any kind.

To comply with the MIT License, you must include the original copyright notice and the license text with any substantial portions of the software you distribute. The simplicity of the MIT License promotes open and unrestricted adoption and reuse of software.

For more detailed information, please review the full [LICENSE](LICENSE) text.
