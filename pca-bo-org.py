#!usr/bin/env python3

"""
"""

__author__ = "Skippybal"
__version__ = "0.1"

import argparse
import json
import math
import os
import sys
import time

import torch

from ioh import get_problem, logger, ProblemClass

import functools
from copy import copy, deepcopy
from typing import Callable, Dict, List, Union

from sklearn.decomposition import PCA
from sklearn.metrics import mean_absolute_percentage_error, r2_score


# from __future__ import annotations

import sys
import numpy as np

from bayes_optim.surrogate import GaussianProcess, RandomForest, trend


from bayes_optim import BO #, ContinuousSpace
# from bayes_optim.extension import PCABO
from bayes_optim.search_space import RealSpace
from bayes_optim.extension import LinearTransform, penalized_acquisition

from bayes_optim.bayes_opt import BO, ParallelBO
from bayes_optim.search_space import RealSpace, SearchSpace
from bayes_optim.surrogate import GaussianProcess, RandomForest, trend
#from bayes_optim.Surrogate import GaussianProcess


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dtype = torch.double
SMOKE_TEST = os.environ.get("SMOKE_TEST")
print(device)

np.random.seed(42)

class PCABO(BO):
    """Dimensionality reduction using Principle Component Decomposition (PCA)

    References

    [RaponiWB+20]
        Raponi, Elena, Hao Wang, Mariusz Bujny, Simonetta Boria, and Carola Doerr.
        "High dimensional bayesian optimization assisted by principal component analysis."
        In International Conference on Parallel Problem Solving from Nature, pp. 169-183.
        Springer, Cham, 2020.

    """

    def __init__(self, n_components: Union[float, int] = None, **kwargs):
        super().__init__(**kwargs)
        if self.model is not None:
            self.logger.warning(
                "The surrogate model will be created automatically by PCA-BO. "
                "The input argument `model` will be ignored"
            )
        assert isinstance(self._search_space, RealSpace)
        self.__search_space = deepcopy(self._search_space)  # the original search space
        self._pca = LinearTransform(n_components=n_components, svd_solver="full", minimize=self.minimize)

    @staticmethod
    def _compute_bounds(pca: PCA, search_space: SearchSpace) -> List[float]:
        C = np.array([(l + u) / 2 for l, u in search_space.bounds])
        radius = np.sqrt(np.sum((np.array([l for l, _ in search_space.bounds]) - C) ** 2))
        C = C - pca.mean_ - pca.center
        C_ = C.dot(pca.components_.T)
        return [(_ - radius, _ + radius) for _ in C_]

    def _create_acquisition(self, fun=None, par=None, return_dx=False, **kwargs) -> Callable:
        acquisition_func = super()._create_acquisition(
            fun=fun, par={} if par is None else par, return_dx=return_dx, **kwargs
        )
        # TODO: make this more general for other acquisition functions
        # wrap the penalized acquisition function for handling the box constraints
        return functools.partial(
            penalized_acquisition,
            acquisition_func=acquisition_func,
            bounds=self.__search_space.bounds,  # hyperbox in the original space
            pca=self._pca,
            return_dx=return_dx,
        )

    def pre_eval_check(self, X: List) -> List:
        """Note that we do not check against duplicated point in PCA-BO since those points are
        sampled in the reduced search space. Please check if this is intended
        """
        if isinstance(X, np.ndarray):
            X = X.tolist()
        return X

    @property
    def xopt(self):
        if not hasattr(self, "data"):
            return None
        fopt = self._get_best(self.data.fitness)
        self._xopt = self.data[np.where(self.data.fitness == fopt)[0][0]]
        return self._xopt

    def ask(self, n_point: int = None) -> List[List[float]]:
        return self._pca.inverse_transform(super().ask(n_point))

    def tell(self, new_X, new_y):
        self.logger.info(f"observing {len(new_X)} points:")
        for i, x in enumerate(new_X):
            self.logger.info(f"#{i + 1} - fitness: {new_y[i]}, solution: {x}")

        index = np.arange(len(new_X))
        if hasattr(self, "data"):
            index += len(self.data)
        # convert `new_X` to a `Solution` object
        new_X = self._to_geno(new_X, index=index, n_eval=1, fitness=new_y)
        self.iter_count += 1
        self.eval_count += len(new_X)

        new_X = self.post_eval_check(new_X)  # remove NaN's
        self.data = self.data + new_X if hasattr(self, "data") else new_X
        # re-fit the PCA transformation
        X = self._pca.fit_transform(np.array(self.data), self.data.fitness)
        bounds = self._compute_bounds(self._pca, self.__search_space)
        # re-set the search space object for the reduced (feature) space
        self._search_space = RealSpace(bounds)
        # update the surrogate model
        self.update_model(X, self.data.fitness)

        # TODO: this line has to commented out on line 176 of extension.py as it couses errros because list concatination
        # self.logger.info(f"xopt/fopt:\n{self.xopt}\n")

    def update_model(self, X: np.ndarray, y: np.ndarray):
        # NOTE: the GPR model will be created since the effective search space (the reduced space
        # is dynamic)
        dim = self._search_space.dim
        bounds = np.asarray(self._search_space.bounds)
        self.model = GaussianProcess(
            mean=trend.constant_trend(dim),
            corr="matern",
            thetaL=1e-3 * (bounds[:, 1] - bounds[:, 0]),
            thetaU=1e3 * (bounds[:, 1] - bounds[:, 0]),
            nugget=1e-6,
            noise_estim=False,
            optimizer="BFGS",
            wait_iter=3,
            random_start=max(10, dim),
            likelihood="concentrated",
            eval_budget=100 * dim,
        )

        _std = np.std(y)
        y_ = y if np.isclose(_std, 0) else (y - np.mean(y)) / _std

        self.fmin, self.fmax = np.min(y_), np.max(y_)
        self.frange = self.fmax - self.fmin

        self.model.fit(X, y_)
        y_hat = self.model.predict(X)

        r2 = r2_score(y_, y_hat)
        MAPE = mean_absolute_percentage_error(y_, y_hat)
        self.logger.info(f"model r2: {r2}, MAPE: {MAPE}")

def create_problem(fid: int, args):
    problem = get_problem(fid, dimension=args.dims, instance=args.instance, problem_class=ProblemClass.BBOB)
    l = logger.Analyzer(
        root="results/PCABO",
        folder_name=args.folder_name,
        algorithm_name=args.algo_name,
        algorithm_info=""
    )
    problem.attach_logger(l)
    return problem, l


def parse_args():
    parser = argparse.ArgumentParser()
    # time = time.
    # parser.add_argument('--folder_name', type=str, default='run')
    # parser.add_argument('--algo_name', type=str, default='DSP')
    # parser.add_argument('--dims', type=int, default=40)
    # parser.add_argument('--func', type=int, default=21)
    # parser.add_argument('--n_trails', type=int, default=5)
    # parser.add_argument('--doe', type=int, default=20)
    # parser.add_argument('--total', type=int, default=400)
    # parser.add_argument('--instance', type=int, default=1)
    # parser.add_argument('--minimize', type=bool, default=True)

    parser.add_argument('--folder_name', type=str, default='run')
    parser.add_argument('--algo_name', type=str, default='pcabo')
    parser.add_argument('--dims', type=int, default=10)
    parser.add_argument('--func', type=int, default=21)
    parser.add_argument('--n_trails', type=int, default=5)
    parser.add_argument('--doe', type=int, default=20)
    parser.add_argument('--total', type=int, default=100)
    parser.add_argument('--instance', type=int, default=1)
    parser.add_argument('--minimize', type=bool, default=True)

    return parser.parse_args()


def main_loop(args, func, minimize=True):
    print(func.bounds)
    dim = 5
    # space = RealSpace([-5, 5]) #* 10
    # space = RealSpace([[-5, 5], [1, 2]])
    # for item
    bound_list = []
    for lb, ub in zip(func.bounds.lb, func.bounds.ub):
        bound_list.append([lb, ub])

    space = RealSpace(bound_list)
    # print(space)
    # space = RealSpace([func.bounds.lb, func.bounds.ub])
    # space = ContinuousSpace([-5, 5]) * dim  # create the search space

    def obj_wrapper(x):
        # print(np.asarray(x))
        res = func(x)
        # print(res)
        # print(func(np.asarray(x)))
        return res #None

    # hyperparameters of the GPR model
    thetaL = 1e-10 * (ub - lb) * np.ones(dim)
    thetaU = 10 * (ub - lb) * np.ones(dim)
    model = GaussianProcess(  # create the GPR model
        thetaL=thetaL, thetaU=thetaU
    )

    opt = PCABO(
        search_space=space,
        obj_fun=obj_wrapper,
        # model=model,
        DoE_size=args.doe,  # number of initial sample points
        max_FEs=args.total,  # maximal function evaluation
        verbose=False,
        n_point=1,
        n_components=0.95,
        acquisition_optimization={"optimizer": "BFGS"},
    )
    opt.run()
    return 0

def main():
    args = parse_args()
    func, _logger = create_problem(args.func, args=args)
    # minimize = True
    for i in range(args.n_trails):
        print(f"\nStarting trail {i}")
        main_loop(args, func, args.minimize)
        func.reset()
        # 2/0
    _logger.close()


if __name__ == '__main__':
    exitcode = main()
    sys.exit(exitcode)