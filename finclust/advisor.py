"""
Portfolio Manager module

This module runs all the steps used.
"""

import warnings
from datetime import datetime, timedelta
from typing import Dict, List, Union

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

from .clustering import Clusterer
from .evaluation import Evaluator
from .utils import calculate_similarities, compose_similarities


class PortfolioManager:
    """
    PortfolioManager is main object of the package.
    
    Parameters
    ----------
    pipeline: Pipeline, optional
        Pipeline object.

    price_column_name: str, default "Close"
        Name of the column with the price values.
    
    verbose: bool, default True
        Whether to log running process info.

    Attributes
    ----------
    returns: pd.DataFrame
        Percentage returns on the asset.
    
    similarities: List[pd.DataFrame]
        Square matrix of pairwise similarities or distances.
    """

    def __init__(self,
                 price_column_name: str = "Close",
                 weights: Dict[str, float] = None,
                 window: timedelta = None,
                 step: timedelta = None,
                 max_return_limit: float = None,
                 
                 pipeline: Pipeline = None,
                 similarity_func: callable = None,
                 clusterer: Clusterer = None,
                 evaluator: Evaluator = None,

                 returns: pd.DataFrame = None,
                 baseline_prices: pd.DataFrame = None,
                 baseline_name: str = "Baseline",
                 
                 evaluate_baseline: bool = True,
                 verbose: bool = True,
                 ):
        self.price_column_name = price_column_name
        self.weights = weights
        self.window = window
        self.step = step
        self.max_return_limit = max_return_limit

        self.pipeline = pipeline
        self.similarity_func = similarity_func
        self.clusterer = clusterer
        self.evaluator = evaluator

        self.returns = returns
        self.baseline_prices = baseline_prices
        self.baseline_name = baseline_name

        self.evaluate_baseline = evaluate_baseline
        self.verbose = verbose

        ## Initialize attributes to None
        self.data = None
        self.output_index = None
        self.similarities = None
        self.clusters = None
        self.baseline_returns = None
        self.baseline_metrics = pd.DataFrame()
        self.portfolios_metrics = None
        
    
    def _log(self, text) -> None:
        """
        Prints actual time and provided text if verbose is True.
        
        Parameters
        ----------
        text: string
            Comment added to printed time.
        """

        if self.verbose:
            print(datetime.now().time().strftime("%H:%M:%S.%f")[:-3], text)
    

    def _check_params(self) -> None:
        need_data = any([p is not None for p in [self.pipeline, self.similarity_func, self.clusterer]])
        if need_data and (self.data is None):
            raise ValueError("Data has to be provided.")


    def _calculate_returns(self, data: Union[pd.DataFrame, pd.Series]) -> pd.DataFrame:
        """
        Fills the returns attribute with the percentage change in prices.

        Parameters
        ----------
        data: pd.DataFrame or pd.Series
            Data with prices in column `self.price_column_name` if MultiIndex.
        """
        if isinstance(data, pd.Series) or data.columns.nlevels == 1:
            prices = data        
        else:
            prices = data[self.price_column_name]
        returns = prices.pct_change().replace([np.inf, -np.inf], 0)
        if self.max_return_limit is not None:
            returns[returns > self.max_return_limit] = self.max_return_limit
        return returns
    

    def _get_portfolio_returns(self, clusters, label: int) -> pd.Series:
        equality = (clusters == label)
        return (self.returns.where(equality, 0).sum(axis=1) / equality.sum(axis=1))


    def _calculate_metrics(self, returns: pd.DataFrame) -> pd.DataFrame:
        metrics = pd.concat(
                [self.evaluator.evaluate(returns.loc[:, col]) for col in returns.columns],
                axis="columns"
            )
        metrics.columns = returns.columns
        return metrics


    def run(self, data: pd.DataFrame = None):
        """
        Runs the selected steps on the provided data.
        
        Parameters
        ----------
        data : array-like
            Data to run the manager.
        
        Returns
        -------
        TODO
        """
        start_time = datetime.now()
        
        ## Creating a copy of the data so that the original data is not corrupted
        if data is not None:
            self.data = data.copy()

        ## Check that the specified parameters are meaningful for the run
        self._check_params()

        ## Preprocessing data
        if self.pipeline is not None:
            self._log("Data preprocessing using pipeline")
            self.data = self.pipeline.fit_transform(self.data)

        if isinstance(self.data.columns, pd.MultiIndex):
            self.data.columns = self.data.columns.remove_unused_levels()
        
        ## Calculate returns
        if self.returns is None:
            self._log("Calculating returns")
            self.returns = self._calculate_returns(self.data)
        
        ## Create output_index
        if self.output_index is None:
            ## Set window and step to the size of the data if None
            if self.window is None:
                self.window = self.data.index[-1] - self.data.index[0]
                self.step = self.window
            self.output_index = pd.date_range(
                start = self.data.index[0] + self.window,
                end = self.data.index[-1],
                freq = self.step,
            )
        starts = self.output_index - self.window
        
        if self.similarity_func is not None:
            ## Calculate similarities
            self._log("Calculating similarities")
            similarities = [calculate_similarities(
                self.data.loc[start:stop],
                func = self.similarity_func,
            ) for start,stop in zip(starts, self.output_index)]
            
            if self.data.columns.nlevels > 1 and len(self.data.columns.levels[0]) > 1:
                ## Compose similarities
                self._log("Composing similarities")
                self.similarities = [compose_similarities(s) for s in similarities]
            else:
                self.similarities = similarities

        if self.clusterer is not None:
            ## Calculate clusters
            self._log("Calculating clusters")
            if self.similarities:
                clusters = [self.clusterer.group(s) for s in self.similarities]
            else:
                clusters = [self.clusterer.group(self.data.loc[start:stop]) for start,stop in zip(starts, self.output_index)]
            self.clusters = pd.DataFrame(
                index = self.returns.index,
            ).join(pd.DataFrame(
                clusters,
                index = self.output_index,
            )).fillna(method="ffill").dropna(axis=0).astype(np.int16)
        
        if self.evaluate_baseline:
            ## Calculate baseline returns
            self._log("Evaluating baseline")
            if self.baseline_prices is not None:
                self.baseline_returns = self._calculate_returns(self.baseline_prices)
            else:
                ones = (~self.returns.isna()).astype(int)
                self.baseline_returns = self._get_portfolio_returns(clusters=ones, label=1)
            if isinstance(self.baseline_returns, pd.Series):
                self.baseline_returns = self.baseline_returns.to_frame(name=self.baseline_name)
        if self.evaluate_baseline and (self.baseline_returns is not None) and self.baseline_metrics.empty:
            ## Evaluate baseline
            self.baseline_metrics = self._calculate_metrics(returns=self.baseline_returns)

        if self.clusters is not None:
            self._log("Calculating returns of portfolios")
            self.portfolios_returns = pd.DataFrame()
            for label in np.unique(self.clusters):
                col = self.clusterer.name
                self.portfolios_returns[f"{col}-{label}"] = self._get_portfolio_returns(clusters=self.clusters, label=label)

            ## Cutting off the nan-values at the beginning
            new_begin = self.portfolios_returns[:self.portfolios_returns.index[0] + self.window].index[-2]
            self.portfolios_returns = self.portfolios_returns[new_begin:]
            ## Set initial returns to 0
            self.portfolios_returns.iloc[0, :] = [0] * self.portfolios_returns.shape[1]

        if self.evaluator is not None:
            ## Evaluate clusters
            self._log("Evaluating cluster portfolios")
            
            self.portfolios_metrics = self._calculate_metrics(returns=self.portfolios_returns)

            if self.evaluate_baseline and not self.baseline_metrics.empty:
                self.portfolios_metrics = self.baseline_metrics.join(self.portfolios_metrics)

        duration = datetime.now() - start_time
        self._log(f"Run completed.\n{'_'*36}\nDuration of the run: {duration}.\n")

        return self.portfolios_metrics
