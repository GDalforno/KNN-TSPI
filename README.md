[![PyPI - Downloads](https://img.shields.io/pypi/dm/knn-tspi)](https://pypistats.org/packages/knn-tspi)
![PyPI - Format](https://img.shields.io/pypi/format/knn-tspi)
![PyPI - Implementation](https://img.shields.io/pypi/implementation/knn-tspi)
[![PyPI - License](https://img.shields.io/pypi/l/knn-tspi)](https://github.com/GDalforno/KNN-TSPI/blob/main/LICENSE.txt)
![PyPI - Status](https://img.shields.io/pypi/status/knn-tspi)
![PyPI - Version](https://img.shields.io/pypi/v/knn-tspi)
[![PEP8](https://img.shields.io/badge/code%20style-pep8-orange.svg)](https://www.python.org/dev/peps/pep-0008/)
![Workflow](https://img.shields.io/github/actions/workflow/status/GDalforno/KNN-TSPI/publish.yaml)
![Release](https://img.shields.io/github/v/release/GDalforno/KNN-TSPI)
![Last Commit](https://img.shields.io/github/last-commit/GDalforno/KNN-TSPI)
![Forks](https://img.shields.io/github/forks/GDalforno/KNN-TSPI)
![Repo Stars](https://img.shields.io/github/stars/GDalforno/KNN-TSPI)

<image alt="logo.png" src="https://raw.githubusercontent.com/GDalforno/KNN-TSPI/main/docs/images/logo.png" width="150rem"/>

K-Nearest Neighbors Time Series Prediction with Invariances (KNN-TSPI) algorithm implementation in python. For details about the model access the [paper](https://www.researchgate.net/publication/300414605_A_Study_of_the_Use_of_Complexity_Measures_in_the_Similarity_Search_Process_Adopted_by_kNN_Algorithm_for_Time_Series_Prediction).

## Installation

### Dependencies

The package depends on the following third party libraries:

- `numpy`

### User Installation

Make sure you have `pip` package manager installed in your environment, then run:

```sh
pip install knn-tspi
```

## Getting Started

Once the package is installed successfully, you can import the `KNeighborsTSPI` class and start forecasting univariate time series in a scikit-learn like manner.

```python
import numpy as np

from knn_tspi import KNeighborsTSPI


data = 0.5 * np.arange(60) + np.random.randn(60)

model = KNeighborsTSPI()
model.fit(data)

y = model.predict(h=5)
```

For more detailed examples on how to use the package access [examples](https://github.com/GDalforno/KNN-TSPI/tree/main/examples).

## Development

### Source Code

Clone the repo with the command:

```
git clone https://github.com/GDalforno/KNN-TSPI.git
```

### Setup

Create a python virtual environment and activate it with the command:

```sh
python3.11 -m venv .venv
source .venv/bin/activate
```

Then, set the enviroment up with the command:

```sh
make setup-dev
```

### Testing

Once the setup is completed, launch the test suite for sanity check with:

```sh
make test-dev
```

### Contributing

You can contributing to the project by opening issues, pull requests and reviewing the code. It will help us a lot if you reference the package on blog posts, articlesm social media, etc.

## License

This project is licensed under the MIT License - see the [License](https://github.com/GDalforno/KNN-TSPI/blob/main/LICENSE.txt) file for details.

## Changelog

See the [changelog](https://github.com/GDalforno/KNN-TSPI/blob/main/CHANGES.txt) for a history of notable changes to knn-tspi.

## Project History

During my research in the field of application of machine learning to forecast time series in 2020, I stumbled with a lack of algorithms and frameworks specialized in this task.

One of my colleagues, [Moisés Rocha](https://github.com/moisesrsantos), send me the paper of a modified KNN for time series prediction along with the experiment code written in MATLAB to help me out with my work. A coupled of days after it, I managed to port the code to both python and R and created this repo to store the resulted files.

Throughtout the years that followed, I have seen a growing interest in this repo, and now, I decided to publish it on pip to make it easier for people to include the model in their time series forecasting toolbox as I did a couple of years ago. As far as I know, there is no other implementation of the KNN-TSPI out there.

I am not planning on creating a CRAN package to distribute the model for the R community anytime soon. With that being said, feel free to implement it yourself if you wish. The core R code can be found [here](https://github.com/GDalforno/KNN-TSPI/tree/main/legacy).

## Communication

- Linkedin: https://www.linkedin.com/in/gabriel-dalforno-7b47ba227/

## References

1. Parmezan, Antonio & Batista, Gustavo. (2015). A Study of the Use of Complexity Measures in the Similarity Search Process Adopted by kNN Algorithm for Time Series Prediction. 45-51. 10.1109/ICMLA.2015.217.
