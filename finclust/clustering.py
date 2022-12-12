"""
Clustering module

Module for performing clustering.
"""

import abc
from typing import Callable, Dict, Union

import pandas as pd


class Clusterer:
    """
    Module for performing clustering.

    Parameters
    ----------
    name: str, default = ""
        Name of the clustering process.
    """

    def __init__(self, name: str = "") -> None:
        self.name = name
    

    @abc.abstractmethod
    def group(self, data: pd.DataFrame) -> Dict[str, int]:
        """
        Groups data to clusters.
        
        Parameters
        ----------
        data: pd.DataFrame
            Data for performing clustering.   
        """
        raise NotImplementedError


class ScikitClusterer(Clusterer):
    """
    Wrapper of clustering methods from Scikit-learn library.

    Parameters
    ----------
    cluster_method: Callable
        Clustering function.
    """

    def __init__(self, cluster_method: Callable, name: str = None) -> None:
        self.cluster_method = cluster_method
        self.name = name if name else cluster_method.__class__.__name__


    def group(self, data: pd.DataFrame) -> Dict[str, int]:
        """
        Calculate cluster based on provided affinities.

        Parameters
        ----------
        data: pd.DataFrame
            Pass the data to the group function
        """
        clustering = self.cluster_method.fit(data)
        symbols = clustering.feature_names_in_ if hasattr(clustering, "feature_names_in_") else data.columns
        labels = clustering.labels_
        return {symbol: label for symbol, label in zip(symbols, labels)}        


class BuyAndHold(Clusterer):
    """
    A simple aggregator that buys everything at the beginning and keeps it until the end.

    Parameters
    ----------
    label: Union[str, int], default=1
        Label for portfolio marking
    """

    def __init__(self, name: str = "Buy&Hold", label: Union[str, int] = 1) -> None:
        super().__init__(name)
        self.label = label

    def group(self, data: pd.DataFrame) -> Dict[str, int]:
        """
        A function that maps all data samples to a given label.

        Parameters
        ----------
        data: pd.DataFrame
            Pass the data to the group function
        
        Returns
        -------
            A dictionary where the keys are asset names and the values are integers
        """
        all_assets = data.columns.levels[1] if data.columns.nlevels > 1 else data.columns
        return {asset: self.label for asset in all_assets}