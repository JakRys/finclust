"""
Clustering module

Module for performing clustering.
"""

import abc
from typing import Callable, Dict
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
        Calculate cluster based on provided similarities.
        """
        clustering = self.cluster_method.fit(data)
        symbols = clustering.feature_names_in_ if hasattr(clustering, "feature_names_in_") else data.columns
        labels = clustering.labels_
        return {symbol: label for symbol, label in zip(symbols, labels)}        
