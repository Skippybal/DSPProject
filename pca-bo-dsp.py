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
from botorch.acquisition import ExpectedImprovement
from botorch.acquisition.analytic import _scaled_improvement, _ei_helper, AnalyticAcquisitionFunction
from botorch.exceptions.warnings import legacy_ei_numerics_warning
from botorch.models.model import Model
from botorch.utils import t_batch_mode_transform

from ioh import get_problem, logger, ProblemClass

import functools
from copy import copy, deepcopy
from typing import Callable, Dict, List, Union

from sklearn.decomposition import PCA
from sklearn.metrics import mean_absolute_percentage_error, r2_score

import torch
from botorch.test_functions import Hartmann
from botorch.models import SingleTaskGP
from gpytorch.mlls.sum_marginal_log_likelihood import ExactMarginalLogLikelihood
from botorch.utils.sampling import draw_sobol_samples
from gpytorch.priors.torch_priors import LogNormalPrior
from sklearn.cross_decomposition import CCA

from gpytorch.kernels import (
    ScaleKernel,
    MaternKernel,
    RBFKernel
)
from gpytorch.priors import (
    NormalPrior,
    GammaPrior,
    LogNormalPrior
)

from botorch.models.transforms import Normalize, Standardize
from botorch.optim import optimize_acqf
from botorch import fit_gpytorch_mll
from botorch.acquisition.logei import qLogNoisyExpectedImprovement
from botorch.exceptions import BadInitialCandidatesWarning
import numpy as np
from gpytorch.constraints.constraints import GreaterThan


# from __future__ import annotations

import sys
import numpy as np

from bayes_optim.surrogate import GaussianProcess, RandomForest, trend


from bayes_optim import BO #, ContinuousSpace
# from bayes_optim.extension import PCABO
from bayes_optim.search_space import RealSpace
from bayes_optim.extension import LinearTransform #, penalized_acquisition

from bayes_optim.bayes_opt import BO, ParallelBO
from bayes_optim.search_space import RealSpace, SearchSpace
from bayes_optim.surrogate import GaussianProcess, RandomForest, trend
from bayes_optim.solution import Solution
from bayes_optim.utils.exception import AskEmptyError, FlatFitnessError
from torch import Tensor

# from __future__ import annotations

import math

from copy import deepcopy
from typing import Optional, Union

import torch

from botorch.acquisition.objective import PosteriorTransform

from botorch.exceptions.warnings import legacy_ei_numerics_warning
from botorch.models.gp_regression import SingleTaskGP
from botorch.models.model import Model

from botorch.utils.transforms import convert_to_target_pre_hook, t_batch_mode_transform
from torch import Tensor


#from bayes_optim.Surrogate import GaussianProcess


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dtype = torch.double
SMOKE_TEST = os.environ.get("SMOKE_TEST")
print(device)

class PenalizedEI(AnalyticAcquisitionFunction):#(ExpectedImprovement):
    def __init__(
            self,
            model: Model,
            best_f: Union[float, Tensor],
            bounds,
            pca,
            return_dx,
            posterior_transform: Optional[PosteriorTransform] = None,
            maximize: bool = True,
    ):
        r"""Single-outcome Expected Improvement (analytic).

        Args:
            model: A fitted single-outcome model.
            best_f: Either a scalar or a `b`-dim Tensor (batch mode) representing
                the best function value observed so far (assumed noiseless).
            posterior_transform: A PosteriorTransform. If using a multi-output model,
                a PosteriorTransform that transforms the multi-output posterior into a
                single-output posterior is required.
            maximize: If True, consider the problem a maximization problem.
        """
        self.return_dx = return_dx
        self.pca = pca
        self.bounds = bounds
        legacy_ei_numerics_warning(legacy_name=type(self).__name__)
        super().__init__(model=model, posterior_transform=posterior_transform)
        self.register_buffer("best_f", torch.as_tensor(best_f))
        self.maximize = maximize

    @t_batch_mode_transform(expected_q=1)
    def forward(self, X: Tensor) -> Tensor:
        r"""Evaluate Expected Improvement on the candidate set X.

        Args:
            X: A `(b1 x ... bk) x 1 x d`-dim batched tensor of `d`-dim design points.
                Expected Improvement is computed for each point individually,
                i.e., what is considered are the marginal posteriors, not the
                joint.

        Returns:
            A `(b1 x ... bk)`-dim tensor of Expected Improvement values at the
            given design points `X`.
        """

        bounds_ = np.atleast_2d(self.bounds)
        # map back the candidate point to check if it falls inside the original domain
        x_ = self.pca.inverse_transform(X)
        idx_lower = np.nonzero(x_ < bounds_[:, 0])[0]
        idx_upper = np.nonzero(x_ > bounds_[:, 1])[0]
        penalty = -1 * (
                np.sum([bounds_[i, 0] - x_[i] for i in idx_lower])
                + np.sum([x_[i] - bounds_[i, 1] for i in idx_upper])
        )
        if penalty == 0:
            mean, sigma = self._mean_and_sigma(X)
            u = _scaled_improvement(mean, sigma, self.best_f, self.maximize)
            # return sigma * _ei_helper(u)
            EI = sigma * _ei_helper(u)
            return EI
        else:
            if self.return_dx:
                # gradient of the penalty in the original space
                g_ = np.zeros((len(x_), 1))
                g_[idx_lower, :] = 1
                g_[idx_upper, :] = -1
                # get the gradient of the penalty in the reduced space
                g = self.pca.components_.dot(g_)
                out = (penalty, g)
            else:
                out = penalty
        return out


def penalized_acquisition(x, acquisition_func, bounds, pca, return_dx):
    bounds_ = np.atleast_2d(bounds)
    # map back the candidate point to check if it falls inside the original domain
    x_ = pca.inverse_transform(x)
    idx_lower = np.nonzero(x_ < bounds_[:, 0])[0]
    idx_upper = np.nonzero(x_ > bounds_[:, 1])[0]
    penalty = -1 * (
        np.sum([bounds_[i, 0] - x_[i] for i in idx_lower])
        + np.sum([x_[i] - bounds_[i, 1] for i in idx_upper])
    )

    if penalty == 0:
        out = acquisition_func(x)
    else:
        if return_dx:
            # gradient of the penalty in the original space
            g_ = np.zeros((len(x_), 1))
            g_[idx_lower, :] = 1
            g_[idx_upper, :] = -1
            # get the gradient of the penalty in the reduced space
            g = pca.components_.dot(g_)
            out = (penalty, g)
        else:
            out = penalty
    return out

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
        # return self._pca.inverse_transform(super().ask(n_point))
        fixed = None

        if self.model is not None: #and self.model.is_fitted:
            n_point = self.n_point if n_point is None else n_point
            msg = f"asking {n_point} points:"
            # 2/0
            X = self.arg_max_acquisition(n_point=n_point, fixed=fixed)
            2 / 0
            X = self.pre_eval_check(X)  # validate the new candidate solutions
            if len(X) < n_point:
                self.logger.warning(
                    f"iteration {self.iter_count}: duplicated solution found "
                    "by optimization! New points is taken from random design."
                )
                N = n_point - len(X)
                # take random samples if the acquisition optimization failed
                X += self.create_DoE(N, fixed=fixed)
        else:  # take the initial DoE
            n_point = self._DoE_size if n_point is None else n_point
            msg = f"asking {n_point} points (using DoE):"
            X = self.create_DoE(n_point, fixed=fixed)

        if len(X) == 0:
            raise AskEmptyError()

        index = np.arange(len(X))
        if hasattr(self, "data"):
            index += len(self.data)

        X = Solution(X, index=index, var_name=self._search_space.var_name)
        self.logger.info(msg)
        for i, _ in enumerate(X):
            self.logger.info(f"#{i + 1} - {self._to_pheno(X[i])}")

        # return self._to_pheno(X)

        return self._pca.inverse_transform(self._to_pheno(X))

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
        # self.model = GaussianProcess(
        #     mean=trend.constant_trend(dim),
        #     corr="matern",
        #     thetaL=1e-3 * (bounds[:, 1] - bounds[:, 0]),
        #     thetaU=1e3 * (bounds[:, 1] - bounds[:, 0]),
        #     nugget=1e-6,
        #     noise_estim=False,
        #     optimizer="BFGS",
        #     wait_iter=3,
        #     random_start=max(10, dim),
        #     likelihood="concentrated",
        #     eval_budget=100 * dim,
        # )

        # mll, model = initialize_model(train_x, train_obj)
        # model, mll =

        _std = np.std(y)
        y_ = y if np.isclose(_std, 0) else (y - np.mean(y)) / _std

        self.fmin, self.fmax = np.min(y_), np.max(y_)
        self.frange = self.fmax - self.fmin

        mll, model = initialize_model(X, y_)
        # model, mll =
        fit_gpytorch_mll(mll, approx_mll=True)
        self.model = model
        # 2/0

        # # self.model.fit(X, y_)
        #
        # # y_hat = self.model.predict(X)
        # # model.training = False
        # # TODO, these ouputs are normalized, so the outputs will be wrong..
        # Also, this methods worsk similar ot saasbo where the prior is so heavy that unless there is good reason to believe a variable is usefull it will not be active
        # so the prediction might become flat real quick
        # y_hat = self.model.posterior(torch.tensor(X))
        # print(y_hat)
        # y_hat = self.model(torch.tensor(X)).loc.detach().numpy()
        # print(y_hat)
        #
        # standar = Standardize(m=1)
        # outs = standar(torch.tensor(y_).reshape(-1,1))[0].detach().numpy()
        # # outs = outs.reshape(-1)
        # print(outs)
        #
        # # 2 / 0
        #
        # # r2 = r2_score(y_, y_hat)
        # # MAPE = mean_absolute_percentage_error(y_, y_hat)
        # r2 = r2_score(outs, y_hat)
        # MAPE = mean_absolute_percentage_error(outs, y_hat)
        # self.logger.info(f"model r2: {r2}, MAPE: {MAPE}")
        # 2 / 0

def initialize_model(train_x, train_obj):
    train_x = torch.tensor(train_x)
    train_obj = torch.tensor(train_obj).reshape(-1,1)
    # print(train_x)
    # 2/0

    covar_mod = RBFKernel(ard_num_dims=train_x.shape[1],
                          lengthscale_prior=LogNormalPrior(loc=np.sqrt(2) + math.log(train_x.shape[1]) * 1/2, scale=np.sqrt(3)),
                          lengthscale_constraint=GreaterThan(1e-4))
    # print(train_obj)

    # define the model for objective
    model = SingleTaskGP(
        train_X=train_x,
        train_Y=train_obj,
        # likelihood=...,
        covar_module=covar_mod,
        input_transform=Normalize(d=train_x.shape[1]),
        outcome_transform=Standardize(m=1)
    ).to(train_x)
    mll = ExactMarginalLogLikelihood(model.likelihood, model)
    return mll, model


def create_problem(fid: int, args):
    problem = get_problem(fid, dimension=args.dims, instance=args.instance, problem_class=ProblemClass.BBOB)
    l = logger.Analyzer(
        root="data",
        folder_name=args.folder_name,
        algorithm_name=args.algo_name,
        algorithm_info=""
    )
    problem.attach_logger(l)
    return problem, l


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--folder_name', type=str, default='run')
    parser.add_argument('--algo_name', type=str, default='DSP-PCABO')
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
    print(space)
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
        DoE_size=20,  # number of initial sample points
        max_FEs=100,  # maximal function evaluation
        verbose=True,
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