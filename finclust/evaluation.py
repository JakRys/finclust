"""
Evaluation module

"""

import abc

import pandas as pd
import quantstats as qs


class Evaluator:
    """
    Modul for evaluation of portfolio returns.

    Parameters
    ----------
    TODO

    Attributes
    ----------
    TODO
    """

    @abc.abstractmethod
    def evaluate(self, returns: pd.DataFrame) -> pd.DataFrame:
        """
        Runs the evaluation of returns.
        """
        raise NotImplementedError


class QuantstatsEvaluator(Evaluator):

    def __init__(self, benchmark=None, rf=0., display=False,
            mode='basic', sep=False, compounded=True,
            periods_per_year=252, prepare_returns=True,
            match_dates=False, **kwargs):
        self.benchmark = benchmark
        self.rf = rf
        self.display = display
        self.mode = mode
        self.sep = sep
        self.compounded = compounded
        self.periods_per_year = periods_per_year
        self.prepare_returns = prepare_returns
        self.match_dates = match_dates
        self.kwargs = kwargs


    def evaluate(self, returns: pd.DataFrame) -> pd.DataFrame:
        return qs.reports.metrics(
            returns,
            benchmark = self.benchmark,
            rf = self.rf,
            display = self.display,
            mode = self.mode,
            sep = self.sep,
            compounded = self.compounded,
            periods_per_year = self.periods_per_year,
            prepare_returns = self.prepare_returns,
            match_dates = self.match_dates,
            **self.kwargs
        )