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
from .utils import calculate_affinities, compose_affinities

import copy


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
    
    affinities: List[pd.DataFrame]
        Square matrix of pairwise affinities or distances.
    """

    def __init__(self,
                 price_column_name: str = "Close",
                 weights: Dict[str, float] = None,
                 window: timedelta = None,
                 step: timedelta = None,
                 max_return_limit: float = None,
                 
                 pipeline: Pipeline = None,
                 affinity_func: callable = None,
                 clusterer: Clusterer = None,
                 evaluator: Evaluator = None,

                 returns: pd.DataFrame = None,
                 asset_weights: pd.DataFrame = None,
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
        self.affinity_func = affinity_func
        self.clusterer = clusterer
        self.evaluator = evaluator

        self.returns = returns
        self.asset_weights = asset_weights
        self.baseline_prices = baseline_prices
        self.baseline_name = baseline_name

        self.evaluate_baseline = evaluate_baseline
        self.verbose = verbose

        ## Initialize attributes to None
        self.data = None
        self.output_index = None
        self.att_affinities = None
        self.affinities = None
        self.clusters = None
        self.baseline_returns = None
        self.baseline_metrics = pd.DataFrame()
        self.portfolios_metrics = None
    

    def copy(self):
        instance = PortfolioManager(
            price_column_name = self.price_column_name,
            weights = self.weights,
            window = self.window,
            step = self.step,
            max_return_limit = self.max_return_limit,

            pipeline = self.pipeline,
            affinity_func = self.affinity_func,
            clusterer = self.clusterer,
            evaluator = self.evaluator,

            returns = self.returns,
            baseline_prices = self.baseline_prices,
            baseline_name = self.baseline_name,

            evaluate_baseline = self.evaluate_baseline,
            verbose = self.verbose,
        )
        instance.data = copy.deepcopy(self.data)
        instance.output_index = copy.deepcopy(self.output_index)
        instance.att_affinities = copy.deepcopy(self.att_affinities)
        instance.affinities = copy.deepcopy(self.affinities)
        instance.clusters = copy.deepcopy(self.clusters)
        instance.baseline_returns = copy.deepcopy(self.baseline_returns)
        instance.baseline_metrics = copy.deepcopy(self.baseline_metrics)
        instance.portfolios_metrics = copy.deepcopy(self.portfolios_metrics)
        return instance
        
    
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
        need_data = any([p is not None for p in [self.pipeline, self.affinity_func, self.clusterer]])
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
        returns = prices.pct_change().replace([np.inf, -np.inf], 0).fillna(0)
        if self.max_return_limit is not None:
            returns[returns > self.max_return_limit] = self.max_return_limit
        return returns
    

    def _get_portfolio_returns(self, clusters, label: int, weights: Union[pd.Series, pd.DataFrame, Dict[str, float]] = None) -> pd.Series:
        returns = self._calculate_returns(self.data)
        if weights is None:
            weights = pd.Series(1, index=clusters.columns)
        elif isinstance(weights, Dict):
            weights = pd.Series(weights)
        
        periods = [(start, stop) for start,stop in zip(clusters.index[:-1], clusters.index[1:])]
        ## We add last period, which is shorter than others
        if clusters.index[-1] != returns.index[-1]:
            periods.append((clusters.index[-1], returns.index[-1]))
        
        portfolio_return = pd.Series(0, index=[periods[0][0]])
        for start,stop in periods:
            ## Get assets for selected period
            assets = clusters.loc[start][clusters.loc[start] == label].index
            cum_return = returns.loc[start:stop, assets].add(1).cumprod()
            ## Select appropriate weights and normalize them 
            if isinstance(weights, pd.DataFrame):
                w = weights.loc[start]
            else:
                w = weights[assets]
            ## Normalize the weights to sum to one
            w /= w.sum()
            ## Calculate cumulative return of the portfolio
            portfolio_value = cum_return.mul(w).sum(axis=1)
            ## Get portfolio returns from its cumulative returns
            portfolio_value = portfolio_value.div(portfolio_value.shift(1)).sub(1)
            portfolio_return = pd.concat([portfolio_return, portfolio_value.iloc[1:]])
        return portfolio_return
        
    
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
        ## Correct dates to existing in the data
        self.output_index = pd.DatetimeIndex([
            (d if d in self.data.index else self.data.loc[self.data.index < d].index[-1]) for d in self.output_index
        ])
        self.output_index = self.output_index.drop_duplicates()
        
        starts = self.output_index - self.window
        
        if (self.affinity_func is not None) and (self.att_affinities is None):
            ## Calculate affinities
            self._log("Calculating affinities")
            self.att_affinities = [calculate_affinities(
                self.data.loc[start:stop],
                func = self.affinity_func,
            ) for start,stop in zip(starts, self.output_index)]
        
        if (self.att_affinities is not None) and (self.affinities is None):
            if self.data.columns.nlevels > 1 and len(self.data.columns.levels[0]) > 1:
                ## Compose affinities
                self._log("Composing affinities")
                self.affinities = [compose_affinities(s) for s in self.att_affinities]
            else:
                self.affinities = self.att_affinities

        if (self.clusterer is not None) and (self.clusters is None):
            ## Calculate clusters
            self._log("Calculating clusters")
            if self.affinities:
                clusters = [self.clusterer.group(s) for s in self.affinities]
            else:
                clusters = [self.clusterer.group(self.data.loc[start:stop]) for start,stop in zip(starts, self.output_index)]
            self.clusters = pd.DataFrame(
                clusters,
                index = self.output_index,
            )
        
        if self.evaluate_baseline:
            ## Calculate baseline returns
            self._log("Evaluating baseline")
            if self.baseline_prices is not None:
                self.baseline_returns = self._calculate_returns(self.baseline_prices)
            else:
                ## Baseline simulation using Buy&Hold strategy of all assets
                tmp_index = [self.returns.index[0], self.returns.index[-1]]
                ones = pd.DataFrame(1, index=tmp_index, columns=self.returns.columns)
                self.baseline_returns = self._get_portfolio_returns(clusters=ones, label=1)
            if isinstance(self.baseline_returns, pd.Series):
                self.baseline_returns = self.baseline_returns.to_frame(name=self.baseline_name)
            ## Align baseline returns by portfolios
            new_begin = self.returns[:self.returns.index[0] + self.window].index[-1]
            self.baseline_returns = self.baseline_returns[new_begin:]
            ## Set initial returns to 0
            self.baseline_returns.iloc[0, :] = 0
            if (self.evaluator is not None) and (self.baseline_returns is not None) and self.baseline_metrics.empty:
                ## Evaluate baseline
                self.baseline_metrics = self._calculate_metrics(returns=self.baseline_returns)

        if self.clusters is not None:
            self._log("Calculating returns of portfolios")
            self.portfolios_returns = pd.DataFrame()
            for label in np.unique(self.clusters):
                name = self.clusterer.name
                col_name = f"{name}-{label}" if name != "" else label
                self.portfolios_returns[col_name] = self._get_portfolio_returns(clusters=self.clusters, label=label, weights=self.asset_weights)

        if self.evaluator is not None:
            ## Evaluate clusters
            self._log("Evaluating cluster portfolios")
            
            self.portfolios_metrics = self._calculate_metrics(returns=self.portfolios_returns)

            if self.evaluate_baseline and not self.baseline_metrics.empty:
                self.portfolios_metrics = self.baseline_metrics.join(self.portfolios_metrics)

        duration = datetime.now() - start_time
        self._log(f"Run completed.\n{'_'*36}\nDuration of the run: {duration}.\n")

        return self.portfolios_metrics
