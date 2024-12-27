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

from botorch.acquisition import MCAcquisitionFunction
from botorch.acquisition.analytic import _scaled_improvement, _ei_helper, AnalyticAcquisitionFunction
from gpytorch.mlls.sum_marginal_log_likelihood import ExactMarginalLogLikelihood
from botorch.utils.sampling import draw_sobol_samples
from gpytorch.priors.torch_priors import LogNormalPrior
from scipy.stats import rankdata, norm
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
from sklearn.decomposition import PCA
from torch import Tensor
from pyDOE import lhs

# from __future__ import annotations

import math

from copy import deepcopy
from typing import Optional, Union, List

import torch

from botorch.acquisition.objective import PosteriorTransform

from botorch.exceptions.warnings import legacy_ei_numerics_warning
from botorch.models.gp_regression import SingleTaskGP
from botorch.models.model import Model

from botorch.utils.transforms import convert_to_target_pre_hook, t_batch_mode_transform
from torch import Tensor

from ioh import get_problem, logger, ProblemClass

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dtype = torch.double
SMOKE_TEST = os.environ.get("SMOKE_TEST")
print(device)

np.random.seed(42)


class LinearTransform(PCA):
    def __init__(self, minimize: bool = True, **kwargs):
        super().__init__(**kwargs)
        self.minimize = minimize

    def fit(self, X: np.ndarray, y: np.ndarray) -> LinearTransform:
        """center the data matrix and scale the data points with respect to the objective values

        Parameters
        ----------
        data : Solution
            the data matrix to scale

        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            the scaled data matrix, the center data matrix, and the objective value
        """
        self.center = X.mean(axis=0)
        X_centered = X - self.center
        y_ = -1 * y if not self.minimize else y
        r = rankdata(y_)
        N = len(y_)
        w = np.log(N) - np.log(r)
        w /= np.sum(w)
        X_scaled = X_centered * w.reshape(-1, 1)
        return super().fit(X_scaled)  # fit the PCA transformation on the scaled data matrix

    def transform(self, X: np.ndarray) -> np.ndarray:
        return super().transform(X - self.center)  # transform the centered data matrix

    def fit_transform(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        self.fit(X, y)
        return self.transform(X)

    def inverse_transform(self, X: np.ndarray) -> np.ndarray:
        if not hasattr(self, "components_"):
            return X
        return super().inverse_transform(X) + self.center


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
        # print(bounds_)
        # map back the candidate point to check if it falls inside the original domain
        # print(X.shape)
        # print(X)

        if X.requires_grad:
            print("---------------------")

        if X.requires_grad:
            x_ = self.pca.inverse_transform(X.detach().numpy())
        else:
            x_ = self.pca.inverse_transform(X.numpy())
        # print(x_.shape)
        # print(x_[0, 0])
        outs = []
        for i in range(x_.shape[0]):
            inst = x_[i, 0]
            # print(inst)
            # print(np.nonzero(inst < bounds_[:, 0]))
            idx_lower = np.nonzero(inst < bounds_[:, 0])[0]
            idx_upper = np.nonzero(inst > bounds_[:, 1])[0]
            # print(idx_lower)
            penalty = -1 * (
                    np.sum([bounds_[i, 0] - x_[i] for i in idx_lower])
                    + np.sum([x_[i] - bounds_[i, 1] for i in idx_upper])
            )
            if penalty == 0:
                mean, sigma = self._mean_and_sigma(X[i])
                u = _scaled_improvement(mean, sigma, self.best_f, self.maximize)
                # return sigma * _ei_helper(u)
                EI = sigma * _ei_helper(u)
                # if X.requires_grad:
                #     print("---------------------")
                print(EI)
                outs.append(EI)
            else:
                # if self.return_dx:
                if X.requires_grad:
                    # gradient of the penalty in the original space
                    g_ = np.zeros((len(inst), 1))
                    g_[idx_lower, :] = 1
                    g_[idx_upper, :] = -1
                    # get the gradient of the penalty in the reduced space
                    g = self.pca.components_.dot(g_)
                    # out = (penalty, g)
                    out = torch.tensor(penalty)
                    out.grad = torch.tensor(g)
                    print(out)
                else:
                    out = torch.tensor(penalty)
                outs.append(out)

        # print(torch.tensor(outs).reshape(-1,1))
        # 2/0
        return torch.tensor(outs) #.reshape(-1,1)
        # idx_lower = np.nonzero(x_ < bounds_[:, 0])[0]
        # idx_upper = np.nonzero(x_ > bounds_[:, 1])[0]
        # print(idx_lower)
        # penalty = -1 * (
        #         np.sum([bounds_[i, 0] - x_[i] for i in idx_lower])
        #         + np.sum([x_[i] - bounds_[i, 1] for i in idx_upper])
        # )
        # if penalty == 0:
        #     mean, sigma = self._mean_and_sigma(X)
        #     u = _scaled_improvement(mean, sigma, self.best_f, self.maximize)
        #     # return sigma * _ei_helper(u)
        #     EI = sigma * _ei_helper(u)
        #     return EI
        # else:
        #     if self.return_dx:
        #         # gradient of the penalty in the original space
        #         g_ = np.zeros((len(x_), 1))
        #         g_[idx_lower, :] = 1
        #         g_[idx_upper, :] = -1
        #         # get the gradient of the penalty in the reduced space
        #         g = self.pca.components_.dot(g_)
        #         out = (penalty, g)
        #     else:
        #         out = penalty
        # return out

# class qPenalizedEI(MCAcquisitionFunction):#(ExpectedImprovement):
#     def __init__(
#         self,
#         bounds,
#         pca,
#         return_dx,
#         model: Model,
#         best_f: Union[float, Tensor],
#         sampler: Optional[MCSampler] = None,
#         objective: Optional[MCAcquisitionObjective] = None,
#         posterior_transform: Optional[PosteriorTransform] = None,
#         X_pending: Optional[Tensor] = None,
#         constraints: Optional[list[Callable[[Tensor], Tensor]]] = None,
#         eta: Union[Tensor, float] = 1e-3,
#     ) -> None:
#         r"""q-Expected Improvement.
#
#         Args:
#             model: A fitted model.
#             best_f: The best objective value observed so far (assumed noiseless). Can be
#                 a scalar, or a `batch_shape`-dim tensor. In case of a batched model, the
#                 tensor can specify different values for each element of the batch.
#             sampler: The sampler used to draw base samples. See `MCAcquisitionFunction`
#                 more details.
#             objective: The MCAcquisitionObjective under which the samples are evaluated.
#                 Defaults to `IdentityMCObjective()`.
#                 NOTE: `ConstrainedMCObjective` for outcome constraints is deprecated in
#                 favor of passing the `constraints` directly to this constructor.
#             posterior_transform: A PosteriorTransform (optional).
#             X_pending:  A `m x d`-dim Tensor of `m` design points that have been
#                 submitted for function evaluation but have not yet been evaluated.
#                 Concatenated into X upon forward call. Copied and set to have no
#                 gradient.
#             constraints: A list of constraint callables which map a Tensor of posterior
#                 samples of dimension `sample_shape x batch-shape x q x m`-dim to a
#                 `sample_shape x batch-shape x q`-dim Tensor. The associated constraints
#                 are considered satisfied if the output is less than zero.
#             eta: Temperature parameter(s) governing the smoothness of the sigmoid
#                 approximation to the constraint indicators. For more details, on this
#                 parameter, see the docs of `compute_smoothed_feasibility_indicator`.
#         """
#         self.return_dx = return_dx
#         self.pca = pca
#         self.bounds = bounds
#         legacy_ei_numerics_warning(legacy_name=type(self).__name__)
#         super().__init__(
#             model=model,
#             sampler=sampler,
#             objective=objective,
#             posterior_transform=posterior_transform,
#             X_pending=X_pending,
#             constraints=constraints,
#             eta=eta,
#         )
#         self.register_buffer("best_f", torch.as_tensor(best_f, dtype=float))
#
#     @t_batch_mode_transform(expected_q=1)
#     def forward(self, X: Tensor) -> Tensor:
#         r"""Evaluate Expected Improvement on the candidate set X.
#
#         Args:
#             X: A `(b1 x ... bk) x 1 x d`-dim batched tensor of `d`-dim design points.
#                 Expected Improvement is computed for each point individually,
#                 i.e., what is considered are the marginal posteriors, not the
#                 joint.
#
#         Returns:
#             A `(b1 x ... bk)`-dim tensor of Expected Improvement values at the
#             given design points `X`.
#         """
#
#         bounds_ = np.atleast_2d(self.bounds)
#         # print(bounds_)
#         # map back the candidate point to check if it falls inside the original domain
#         # print(X.shape)
#         # print(X)
#
#         if X.requires_grad:
#             print("---------------------")
#
#         if X.requires_grad:
#             x_ = self.pca.inverse_transform(X.detach().numpy())
#         else:
#             x_ = self.pca.inverse_transform(X.numpy())
#         # print(x_.shape)
#         # print(x_[0, 0])
#         outs = []
#         for i in range(x_.shape[0]):
#             inst = x_[i, 0]
#             # print(inst)
#             # print(np.nonzero(inst < bounds_[:, 0]))
#             idx_lower = np.nonzero(inst < bounds_[:, 0])[0]
#             idx_upper = np.nonzero(inst > bounds_[:, 1])[0]
#             # print(idx_lower)
#             penalty = -1 * (
#                     np.sum([bounds_[i, 0] - x_[i] for i in idx_lower])
#                     + np.sum([x_[i] - bounds_[i, 1] for i in idx_upper])
#             )
#             if penalty == 0:
#                 mean, sigma = self._mean_and_sigma(X[i])
#                 u = _scaled_improvement(mean, sigma, self.best_f, self.maximize)
#                 # return sigma * _ei_helper(u)
#                 EI = sigma * _ei_helper(u)
#                 # if X.requires_grad:
#                 #     print("---------------------")
#                 print(EI)
#                 outs.append(EI)
#             else:
#                 if self.return_dx:
#                 # if X.requires_grad:
#                     # gradient of the penalty in the original space
#                     g_ = np.zeros((len(x_), 1))
#                     g_[idx_lower, :] = 1
#                     g_[idx_upper, :] = -1
#                     # get the gradient of the penalty in the reduced space
#                     g = self.pca.components_.dot(g_)
#                     out = (penalty, g)
#                 else:
#                     out = torch.tensor(penalty)
#                 outs.append(out)
#
#
#         return torch.tensor(outs) #.reshape(-1,1)


# def optimize_acqf_and_get_observation(func, acq_func, num_restarts=20, raw_samples=512, NOISE_SE=0.05):
def optimize_acqf_and_get_observation(func, acq_func, bounds, num_restarts=20, raw_samples=512):
    """Optimizes the acquisition function, and returns a new candidate and a noisy observation."""
    # optimize
    # candidates, _ = optimize_acqf(
    #     acq_function=acq_func,
    #     bounds=func.bounds,
    #     q=1,
    #     num_restarts=num_restarts,
    #     raw_samples=raw_samples,  # used for intialization heuristic
    #     options={"batch_limit": 5, "maxiter": 200},
    # )

    start = time.time()
    lb, ub = zip(*bounds)

    candidates, _ = optimize_acqf(
        acq_function=acq_func,
        bounds=torch.tensor(np.array([lb, ub])), #.to(device),
        q=1,#5,
        num_restarts=15,
        raw_samples=256,  # used for intialization heuristic
        retry_on_optimization_warning=False,
        options={"batch_limit": 64, "maxiter": 100,
                 "nonnegative": False,
                 "sample_around_best": True,
                 "sample_around_best_sigma": 0.1},
        # sequential=True
        # options={"batch_limit": 64, "maxiter": 300,
        #          "nonnegative": False,
        #          "sample_around_best": True,
        #          "sample_around_best_sigma":0.1},
    )

    # observe new values
    new_x = candidates.detach()
    # exact_obj = func(new_x).unsqueeze(-1)  # add output dimension
    exact_obj = func(new_x.cpu().numpy())
    # print(exact_obj)
    # 2/0
    # print("''''''''")
    # print(exact_obj)
    # print(func.evaluate_true(new_x))
    # train_obj = exact_obj + NOISE_SE * torch.randn_like(exact_obj)

    end = time.time()
    print(end - start)

    return new_x, exact_obj#train_obj


def initialize_model(train_x, train_obj, indices=None):

    covar_mod = RBFKernel(ard_num_dims=train_x.shape[1],
                          lengthscale_prior=LogNormalPrior(loc=np.sqrt(2) + math.log(train_x.shape[1]) * 1/2, scale=np.sqrt(3)),
                          lengthscale_constraint=GreaterThan(1e-4))
    # print(train_obj)

    # define the model for objective
    model = SingleTaskGP(
        train_X=train_x,
        train_Y=train_obj,
        # train_Yvar=train_yvar.expand_as(train_obj),
        # likelihood=...,
        covar_module=covar_mod,
        # input_transform=Normalize(train_x.shape[1], [0,1]),
        input_transform=Normalize(d=train_x.shape[1]),
        outcome_transform=Standardize(m=1)
    ).to(train_x)
    mll = ExactMarginalLogLikelihood(model.likelihood, model)
    #mll.to(device)
    return mll, model



# def create_problem(fid: int, dims, instance, args):
    # problem = get_problem(fid, dimension=dims, instance=instance, problem_class=ProblemClass.BBOB)
    # l = logger.Analyzer(
    #     root="data",
    #     folder_name="run",
    #     algorithm_name="DSP",
    #     algorithm_info=""
    # )
    # problem.attach_logger(l)
    # return problem, l
def create_problem(fid: int, args):
    problem = get_problem(fid, dimension=args.dims, instance=args.instance, problem_class=ProblemClass.BBOB)
    l = logger.Analyzer(
        root="DSP-PCABO",
        folder_name=args.folder_name,
        algorithm_name=args.algo_name,
        algorithm_info=""
    )
    problem.attach_logger(l)
    return problem, l


def parse_args():
    parser = argparse.ArgumentParser()
    # time = time.
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


def _compute_bounds(pca: PCA, search_space: SearchSpace) -> List[float]:
    C = np.array([(l + u) / 2 for l, u in search_space.bounds])
    radius = np.sqrt(np.sum((np.array([l for l, _ in search_space.bounds]) - C) ** 2))
    C = C - pca.mean_ - pca.center
    C_ = C.dot(pca.components_.T)
    return [(_ - radius, _ + radius) for _ in C_]


def main_loop(args, func, minimize=True):

    n = args.doe #20 #20#10#14


    train_x = draw_sobol_samples(
        bounds=torch.tensor(np.array([func.bounds.lb, func.bounds.ub])), n=n, q=1, seed=torch.randint(0, 10000, (1,)).item()
    ).squeeze(1)
    # exact_obj = func(train_x).unsqueeze(-1)  # add output dimension
    exact_obj = func(train_x.cpu().numpy())#.unsqueeze(-1)

    # best_observed_value = exact_obj.max().item()
    best_observed_value = max(exact_obj)
    train_obj = exact_obj #+ NOISE_SE * torch.randn_like(exact_obj)


    # best_observed = [train_obj[0].item()]
    best_observed = [train_obj[0]]
    model = None


    # for i in range(1, train_obj.shape[0]):
    for i in range(1, len(train_obj)):
        if train_obj[i] < best_observed[-1]:
            best_observed.append(train_obj[i])
        else:
            best_observed.append(best_observed[-1])

    # best_observed = [best_observed_value]
    # if minimize:
    #     train_obj = [-i for i in train_obj]

    train_obj = torch.tensor(np.array(train_obj).reshape(-1,1)) #.to(device)

    # mll, model = initialize_model(train_x, train_obj)
    # 2/0
    N_BATCH = args.total-args.doe#90#40

    bound_list = []
    for lb, ub in zip(func.bounds.lb, func.bounds.ub):
        bound_list.append([lb, ub])

    space = RealSpace(bound_list)

    # for iteration in range(1, ((N_BATCH + 1)//5)+1):
    for iteration in range(1, N_BATCH + 1 + 1):
        # fit the models

        _pca = LinearTransform(n_components=0.95, svd_solver="full", minimize=args.minimize)
        X = _pca.fit_transform(np.array(train_x), np.array(train_obj))
        current_bounds = _compute_bounds(_pca, space)
        # print(X)
        # print(current_bounds)
        # re-set the search space object for the reduced (feature) space
        # self._search_space = RealSpace(bounds)
        # 2/0
        # TODO: negative train objective becuase we are
        mll, model = initialize_model(torch.tensor(X), train_obj)

        start = time.time()
        fit_gpytorch_mll(mll, approx_mll=True)
        end = time.time()
        # print(end - start)
        # ei = qLogNoisyExpectedImprovement(model=model, X_baseline=train_x)
        # ei = PenalizedEI(model=model, pca=_pca, bounds=bound_list, best_f=min(train_obj), return_dx=False, maximize=True)
        # ei = 1PenalizedEI(model=model, pca=_pca, bounds=bound_list, best_f=min(train_obj), return_dx=False,
        #                  maximize=True)

        # # optimize and get new observation
        # new_x, new_obj = optimize_acqf_and_get_observation(func, ei, bounds=current_bounds)
        # print(bound_list)
        bound_arr = np.array(current_bounds)
        # print(bound_arr)
        diffs = bound_arr[:,1] - bound_arr[:,0]
        # print(diffs)
        D = lhs(len(current_bounds), 20000) # * 2 * box_size - box_size
        # print(D)
        # print(diffs)
        # D = D @ diffs.reshape(-1,1) + bound_arr[:,0]
        # print(D)
        D = D * diffs + bound_arr[:,0]
        # print(D)

        # f = np.array([[  0.,   1.,   4.],[  0.,   4.,  10.],[  0.,   7.,  16.]])
        # print(f * np.array([0,2,0]) + np.array([2,2,0]))
        # print(bound_arr[:,0])

        v = model(torch.tensor(D))
        # var = model.likelihood(v)
        f_mean = v.mean
        f_var = v.variance

        def penalizedEI(x, model, bounds, pca, f_min):
            bounds_ = np.atleast_2d(bounds)
            # print(bounds_)
            # map back the candidate point to check if it falls inside the original domain
            # print(X.shape)
            # print(X)

            v = model(torch.tensor(x))
            f_mean = v.mean #.detach().numpy()
            f_var = v.variance #.detach().numpy()
            # print(v)
            # print(f_mean)
            x_ = pca.inverse_transform(x)
            # print(x_.shape)
            # print(x_[0, 0])
            outs = []
            for i in range(x_.shape[0]):
                inst = x_[i]
                # print(inst)
                # print(torch.tensor(inst))
                # print(inst)

                idx_lower = np.nonzero(inst < bounds_[:, 0])[0]
                idx_upper = np.nonzero(inst > bounds_[:, 1])[0]

                penalty = -1 * (
                        np.sum([bounds_[j, 0] - inst[j] for j in idx_lower])
                        + np.sum([inst[j] - bounds_[j, 1] for j in idx_upper])
                )
                # print(penalty)

                # print(x[i])
                # print(torch.tensor(x[i]).shape)

                if penalty == 0:
                    # print("Fair")
                    # v = model(torch.tensor(x[i]))
                    # f_mean = v.mean.detach().numpy()
                    # f_var = v.variance.detach().numpy()
                    # print(f_var[i])
                    # ei = np.zeros((D_size, 1))
                    std_dev = np.sqrt(f_var[i].item())
                    if f_var[i].item() != 0:
                        # z = (f_mean[i].item() - f_max) / std_dev
                        # out = (f_mean[i].item() - f_max) * norm.cdf(z) + std_dev * norm.pdf(z)
                        z = (f_min - f_mean[i].item()) / std_dev
                        out = (f_min - f_mean[i].item()) * norm.cdf(z) + std_dev * norm.pdf(z)
                    else:
                        out = 0

                    outs.append(out)
                else:
                    # print("Pen")
                    # print(penalty)
                    out = torch.tensor(penalty)
                    outs.append(out)
            return torch.tensor(outs)

        # TODO: check this, as im pretty sure we are minimizing, but the EI funciton here is for maximizing....
        # ei_d = EI(len(D), max(f_s), f_mean.detach().numpy(), f_var.detach().numpy())
        # ei_d = EI(len(D), min(f_s), -f_mean.detach().numpy(), f_var.detach().numpy())
        # index = np.argmax(ei_d)
        ei_d = penalizedEI(D, model, bound_list, _pca, min(train_obj))
        index = np.argmax(ei_d)
        # print(ei_d)

        candidate = _pca.inverse_transform(D[index]).clip(func.bounds.lb, func.bounds.ub)
        # print(candidate)

        new_obj = func(candidate)
        # print(new_obj)
        # 2/0

        # self._pca.inverse_transform(super().ask(n_point))

        train_x = torch.cat([train_x, torch.tensor([candidate])])
        # print(train_x)
        # print(torch.tensor([new_obj]))
        train_obj = torch.cat([train_obj, torch.tensor([[new_obj]])])
        # print(train_obj)
        # if minimize:
        #     train_obj = torch.cat([train_obj, torch.tensor(-np.array(new_obj).reshape(-1,1))]) #.to(device)
        # else:
        #     train_obj = torch.cat([train_obj, torch.tensor(np.array(new_obj).reshape(-1,1))])#.to(device)


        # if new_obj[0] < best_observed[-1]:
        #     best_observed.append(new_obj[0])
        #
        # else:
        #     best_observed.append(best_observed[-1])
        #
        #
        # indices = None


        # mll, model = initialize_model(train_x, train_obj, indices)

        if iteration % 2 == 0:
            print(".")

    return 0

def main():
    args = parse_args()
    func, _logger = create_problem(args.func, args=args)
    # minimize = True
    for i in range(args.n_trails):
        print(f"\nStarting trail {i}")
        main_loop(args, func, args.minimize)
        func.reset()
    _logger.close()


if __name__ == '__main__':
    exitcode = main()
    sys.exit(exitcode)