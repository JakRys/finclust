# FinClust

[![Open Source? Yes!](https://badgen.net/badge/Open%20Source%20%3F/Yes%21/blue?icon=github)](https://github.com/Naereen/badges/)
[![Imports: isort](https://img.shields.io/badge/%20imports-isort-%231674b1?style=flat&labelColor=ef8336)](https://pycqa.github.io/isort/)

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

- TBA,
- ...,
- visualizations.


The framework is designed for easy customization and extension of its functionality.


## Installation

```bash
python -m pip install git+https://github.com/JakRys/finclust
```
or (not yet available):
```bash
pip install finclust
```

## Features
See the `README` in the [finclust folder](finclust).


## Usage
It is simple to use this package. After the import, you need to do three steps:

1. Create _PortfolioManager_ (according to how you want to evaluate);
2. Run the **evaluation**.
3. Create **visualizations**.

```python
from sklearn.pipeline import Pipeline

from finclust.??? import TODO


```
See the [examples folder](examples) for more details.


## License
TBA

## Disclaimer
This library was created as part of my master's thesis.

The structure is heavily inspired by the [SeqRep package](https://github.com/MIR-MU/seqrep) (my earlier project), which is designed for supervised learning.


## Acknowledgement

First of all, I would like to thank Petr Sojka for supervising my thesis. I am grateful to [Michal Stefanik](https://github.com/stefanik12) for his valuable consultations, especially regarding the code. Gratitude also belongs to all members of the [MIR-MU](https://github.com/MIR-MU/) group for their comments.

## How to cite
```
@mastersthesis{Rysavy2022thesis,
  author  = {Ryšavý, Jakub},
  title   = {Machine Learning for Algorithmic Trading of Decentralized Finances},
  school  = {Masarykova univerzita, Fakulta informatiky},
  location = {Brno},
  supervisor = {Petr Sojka},
  year    = {2022},
  type    = {Master's thesis},
  url     = {},
  url_date = {},
}
```



[![](https://img.shields.io/badge/back%20to%20top-%E2%86%A9-blue)](#finclust)
