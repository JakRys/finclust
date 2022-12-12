# FinClust

[![Open Source? Yes!](https://badgen.net/badge/Open%20Source%20%3F/Yes%21/blue?icon=github)](https://github.com/Naereen/badges/)
[![GitHub license](https://img.shields.io/github/license/Naereen/StrapDown.js.svg)](LICENSE)
[![Imports: isort](https://img.shields.io/badge/%20imports-isort-%231674b1?style=flat&labelColor=ef8336)](https://pycqa.github.io/isort/)
[![CodeFactor](https://www.codefactor.io/repository/github/jakrys/finclust/badge)](https://www.codefactor.io/repository/github/jakrys/finclust)
[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/JakRys/finclust/blob/main/examples/simple_example.ipynb)



_Library for portfolio creation (by clustering) of financial time series with evaluation and visualizations._

## Table of Content
<details>
<summary>Click to expand!</summary>

- [Table of Content](#table-of-content)
- [Description](#description)
- [Installation](#installation)
- [Features](#features)
- [Usage](#usage)
- [License](#license)
- [Disclaimer](#disclaimer)
- [Acknowledgement](#acknowledgement)
- [How to cite](#how-to-cite)
</details>


## Description

This library aims to simplify the workflow of **portfolio creation and evaluation**. It is primarily focused on financial time series data. It helps with:

- **data** preprocessing,
- calculation of **affinities**,
- **clustering** of assets/portfolio selection,
- calculation and **evaluation** of portfolios returns,
- visualizations.


The framework is designed for **easy customization** and extension of its functionality.


## Installation

```bash
python -m pip install git+https://github.com/JakRys/finclust
```


## Usage
It is simple to use this package. After the import, you need to do three steps:

1. Create _PortfolioManager_ (according to how you want to evaluate);
2. Run the **evaluation**.
3. Create **visualizations**.

```python
## Import the required modules
from datetime import timedelta
import numpy as np

from finclust import PortfolioManager
from finclust.clustering import ScikitClusterer
from finclust.evaluation import QuantstatsEvaluator

from sklearn.cluster import AgglomerativeClustering

## Create instance of PortfolioManager
mgr = PortfolioManager(
    window = timedelta(weeks=16),
    step = timedelta(weeks=4),
    affinity_func = np.corrcoef,
    clusterer = ScikitClusterer(
        cluster_method = AgglomerativeClustering(affinity="precomputed", linkage="single", n_clusters=5),
    ),
    evaluator = QuantstatsEvaluator(),
)
## Run the process
mgr.run(data=data)
```
See the [examples folder](examples) for more details or [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/JakRys/finclust/blob/main/examples/simple_example.ipynb).


## License
[![GitHub license](https://img.shields.io/github/license/Naereen/StrapDown.js.svg)](LICENSE)

This package is licensed under the [MIT license](LICENSE), so it is open source.

## Disclaimer
This library was created as part of [my master's thesis](https://is.muni.cz/th/xc3yt/).

The structure is heavily inspired by the [SeqRep package](https://github.com/MIR-MU/seqrep) (my earlier project), which is designed for supervised learning.


## Acknowledgement

First of all, I would like to thank Petr Sojka for supervising my thesis. I am grateful to [Michal Stefanik](https://github.com/stefanik12) for his valuable consultations, especially regarding the code. Gratitude also belongs to all members of the [MIR-MU](https://github.com/MIR-MU/) group for their comments.

## How to cite
```
@mastersthesis{Rysavy2022thesis,
  AUTHOR = {Ryšavý, Jakub},
  TITLE = {Machine Learning for Algorithmic Trading of Decentralized Finances},
  YEAR = {2022},
  TYPE = {Master's thesis},
  INSTITUTION = {Masaryk University, Faculty of Informatics},
  LOCATION = {Brno},
  SUPERVISOR = {Petr Sojka},
  URL = {https://is.muni.cz/th/xc3yt/},
  URL_DATE = {2022-12-13},
}
```


[![](https://img.shields.io/badge/back%20to%20top-%E2%86%A9-blue)](#finclust)
