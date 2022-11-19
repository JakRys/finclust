"""
Visualization module

"""

import abc
from typing import List, Union
import quantstats as qs
from . import PortfolioManager


class Visualizator:
    """
    Main class of Visualizator module.
    """
    def _check_parameter(self, mgr: PortfolioManager, param: str):
        if getattr(mgr, param) is None:
            raise ValueError(f"It is not possible to create a visualization because the `{param}` parameter is None.")

    @abc.abstractmethod
    def visualize(self, mgr: PortfolioManager):
        """
        Create visualizations.

        Parameters
        ----------
        mgr: PortfolioManager
            Instance of PortfolioManager with desired attributes.
        """
        raise NotImplementedError


class SequentialVisualizator(Visualizator):
    """
    Wrapper for the visualizators in the list

    Attributes
    ----------
    visualizators_list: List[Visualizator]
        List of visualizators
    """
    def __init__(self, visualizators_list: List[Visualizator]) -> None:
        self.visualizators_list = visualizators_list

    def visualize(self, mgr: PortfolioManager):
        """
        Runs the visualize method on all elements.
        
        Parameters
        ----------
        mgr: PortfolioManager
            Pass the portfolio manager to the visualizators
        
        Returns
        -------
            Output of the visualize method of the last element of the visualizators_list
        """
        for vis in self.visualizators_list[:-1]:
            vis.visualize(mgr)
        return self.visualizators_list[-1].visualize(mgr)


class CumulativeReturnsVisualizator(Visualizator):
    """
    Object that creates a plot of cumulative returns.
    """
    def __init__(self, plotting_backend: str = None, title: str = "Returns of portfolios",
                 include_baseline: bool = True, **kwargs) -> None:
        self.plotting_backend = plotting_backend
        self.title = title
        self.include_baseline = include_baseline
        self.kwargs = kwargs

    def visualize(self, mgr: PortfolioManager):
        """
        Creates a plot of cumulative returns
        """
        self._check_parameter(mgr=mgr, param="portfolios_returns")
        if self.include_baseline and mgr.baseline_returns is not None:
            returns_ = mgr.baseline_returns.join(mgr.portfolios_returns)
        else:
            returns_ = mgr.portfolios_returns
        return ((1 + returns_).cumprod() - 1).plot(
            backend = self.plotting_backend,
            title = self.title,
            **self.kwargs,
        )


class QuantstatsVisualizator(Visualizator):
    """
    Wrapper of Quantstats' plot method.
    """
    def __init__(self, benchmark=None, grayscale=False,
            figsize=(8, 5), mode='basic', compounded=True,
            periods_per_year=252, prepare_returns=True,
            match_dates=False, strategy_name: Union[str, int] = None):
        self.benchmark = benchmark
        self.grayscale = grayscale
        self.figsize = figsize
        self.mode = mode
        self.compounded = compounded
        self.periods_per_year = periods_per_year
        self.prepare_returns = prepare_returns
        self.match_dates = match_dates

        self.strategy_name = strategy_name


    def visualize(self, mgr: PortfolioManager):
        self._check_parameter(mgr=mgr, param="portfolios_returns")
        if self.strategy_name is None:
            column = mgr.portfolios_returns.columns[0]
        elif isinstance(self.strategy_name, int) and (self.strategy_name < len(mgr.portfolios_returns.columns)):
            column = mgr.portfolios_returns.columns[self.strategy_name]
        elif isinstance(self.strategy_name, str) and (self.strategy_name in mgr.portfolios_returns.columns):
            column = self.strategy_name
        else:
            raise ValueError("The strategy_name parameter is not set correctly.")

        return qs.reports.plots(
            mgr.portfolios_returns[column],
            benchmark = self.benchmark,
            grayscale = self.grayscale,
            figsize = self.figsize,
            mode = self.mode,
            compounded = self.compounded,
            periods_per_year = self.periods_per_year,
            prepare_returns = self.prepare_returns,
            match_dates = self.match_dates,
        )
