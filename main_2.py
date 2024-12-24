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
from matplotlib import pyplot as plt
from gpytorch.constraints.constraints import GreaterThan

# from cca_zoo.data.simulated import LinearSimulatedData
from cca_zoo.model_selection import GridSearchCV
from cca_zoo.nonparametric import KCCA

from SALib.analyze.sobol import analyze
from SALib.sample.sobol import sample
from SALib.sample import saltelli, sobol

# from Extra.bench import BenchSuiteFunction
from Extra.mujo import MujocoFunction
from Extra.synth import Embedded
from scipy.stats import sobol_indices, uniform

import interpret.glassbox
import shap
import pandas as pd

from ioh import get_problem, logger, ProblemClass

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dtype = torch.double
SMOKE_TEST = os.environ.get("SMOKE_TEST")
print(device)


# def optimize_acqf_and_get_observation(func, acq_func, num_restarts=20, raw_samples=512, NOISE_SE=0.05):
def optimize_acqf_and_get_observation(func, acq_func, num_restarts=20, raw_samples=512):
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

    candidates, _ = optimize_acqf(
        acq_function=acq_func,
        bounds=torch.tensor(np.array([func.bounds.lb, func.bounds.ub])), #.to(device),
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
        root="data",
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
    parser.add_argument('--algo_name', type=str, default='DSP')
    parser.add_argument('--dims', type=int, default=40)
    parser.add_argument('--func', type=int, default=21)
    parser.add_argument('--n_trails', type=int, default=5)
    parser.add_argument('--doe', type=int, default=20)
    parser.add_argument('--total', type=int, default=400)
    parser.add_argument('--instance', type=int, default=1)
    parser.add_argument('--minimize', type=bool, default=True)

    return parser.parse_args()


def main_loop(args, func, minimize=True):
    # torch.manual_seed(0)

    # func_org = Hartmann(dim=6, negate=True)
    # # func_tup = (Embedded, dict(function=Hartmann(dim=6), noise_std=0.01, negate=True, dim=25))
    # func_tup = (Embedded, dict(function=Hartmann(dim=6), noise_std=0.01, negate=True, dim=1000))
    # # func_tup = (Embedded, dict(function=Hartmann(dim=6), noise_std=0.01, negate=True, dim=100))
    # # func_tup = (BenchSuiteFunction, dict(negate=True, task_id='mopta'))
    # # func_tup = (MujocoFunction, dict(negate=True, bounds=[[-1,1] * 888] , container='mujoco', task_id='ant'))
    # func = func_tup[0](**func_tup[1])

    # func, _logger = create_problem(21, dims=40, instance=1)
    # minimize = True
    # print(func.bounds)

    # NOISE_SE = 0.05
    # train_yvar = torch.tensor(NOISE_SE ** 2, device=device, dtype=dtype)

    # bounds = torch.tensor([[0.0] * 6, [1.0] * 6], device=device, dtype=dtype)
    # print(bounds)

    n = args.doe #20 #20#10#14
    # generate initial training data
    # print(func.bounds)
    # print(torch.tensor([func.bounds.lb, func.bounds.ub]))
    # print(Hartmann(dim=6, negate=True).bounds)
    # train_x = draw_sobol_samples(
    #     bounds=func.bounds, n=n, q=1, seed=torch.randint(0, 10000, (1,)).item()
    # # ).squeeze(1)
    # train_x = draw_sobol_samples(
    #     bounds=torch.tensor(np.array([func.bounds.lb, func.bounds.ub])).to(device), n=n, q=1, seed=torch.randint(0, 10000, (1,)).item()
    # ).squeeze(1).to(device)
    # # exact_obj = func(train_x).unsqueeze(-1)  # add output dimension
    # exact_obj = func(train_x.cpu().numpy())#.unsqueeze(-1)
    # # print(exact_obj)
    # # print(train_x)

    train_x = draw_sobol_samples(
        bounds=torch.tensor(np.array([func.bounds.lb, func.bounds.ub])), n=n, q=1, seed=torch.randint(0, 10000, (1,)).item()
    ).squeeze(1)
    # exact_obj = func(train_x).unsqueeze(-1)  # add output dimension
    exact_obj = func(train_x.cpu().numpy())#.unsqueeze(-1)

    # best_observed_value = exact_obj.max().item()
    best_observed_value = max(exact_obj)
    train_obj = exact_obj #+ NOISE_SE * torch.randn_like(exact_obj)
    # print(best_observed_value)
    # print(train_obj)
    # print(train_x.shape)
    # print(train_x.shape[1])

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
    if minimize:
        train_obj = [-i for i in train_obj]

    train_obj = torch.tensor(np.array(train_obj).reshape(-1,1)) #.to(device)

    mll, model = initialize_model(train_x, train_obj)
    # 2/0
    N_BATCH = args.total-args.doe#90#40

    # for iteration in range(1, ((N_BATCH + 1)//5)+1):
    for iteration in range(1, N_BATCH + 1 + 1):
        # fit the models
        start = time.time()
        fit_gpytorch_mll(mll, approx_mll=True)
        end = time.time()
        print(end - start)
        ei = qLogNoisyExpectedImprovement(model=model, X_baseline=train_x)

        # optimize and get new observation
        new_x, new_obj = optimize_acqf_and_get_observation(func, ei)
        # print(new_obj)

        # update training points
        # train_x = torch.cat([train_x, new_x])
        # if minimize:
        #     train_obj = torch.cat([train_obj, torch.tensor(-np.array(new_obj).reshape(-1,1)).to(device)]) #.to(device)
        # else:
        #     train_obj = torch.cat([train_obj, torch.tensor(np.array(new_obj).reshape(-1,1)).to(device)])#.to(device)

        train_x = torch.cat([train_x, new_x])
        if minimize:
            train_obj = torch.cat([train_obj, torch.tensor(-np.array(new_obj).reshape(-1,1))]) #.to(device)
        else:
            train_obj = torch.cat([train_obj, torch.tensor(np.array(new_obj).reshape(-1,1))])#.to(device)

        # if minimize is True:
        #     train_obj = -train_obj

        # update progress
        # TODO: This only works if we assume no noise
        # best_value = func(train_x).max().item()
        # TODO: Ask how to handle this

        if new_obj[0] < best_observed[-1]:
            best_observed.append(new_obj[0])
            # print(new_obj)
            # print(iteration + n)
            # # print(np.log10(func_org.optimal_value - new_obj.item()))
            # print(new_obj[0])
        else:
            best_observed.append(best_observed[-1])

        # best_value = func(train_x).min().item()
        # best_observed.append(best_value)
        # if i > 1:
        # indices = get_sobol(model, train_x, train_obj)

        # indices = get_sobol_2(model, train_x, train_obj)
        indices = None

        # if (iteration+1) % 10 == 0:
        #     get_shaps(train_x, train_obj)

        # get_cca(model, train_x, train_obj)
        # print(indices)
        mll, model = initialize_model(train_x, train_obj, indices)
        # print(model.state_dict())
        # mll, model = initialize_model(train_x, train_obj)

        # print(".", end="")
        if iteration % 2 == 0:
            print(".")
        # if iteration % 10 == 0:
        #     print(".")
    # print(best_observed_value)
    # print(best_observed)
    # GLOBAL_MAXIMUM = func_org.optimal_value #+ 0.1 # Add possible noise
    # print(GLOBAL_MAXIMUM)

    # GLOBAL_MAXIMUM = func.optimum.y #.optimum.y
    # print(GLOBAL_MAXIMUM)
    #
    # # iters = np.arange(N_BATCH + 1)
    # iters = np.arange(start=1, stop=N_BATCH + n+1)
    # # y_ei = np.log10(GLOBAL_MAXIMUM - np.asarray(best_observed))
    # # y_ei = np.log10(GLOBAL_MAXIMUM - np.asarray(best_observed))
    #
    # # y_ei = np.log(GLOBAL_MAXIMUM - np.asarray(best_observed))
    # # y_ei = np.log10(GLOBAL_MAXIMUM - np.asarray(best_observed))
    # y_ei = np.asarray(best_observed)
    # print(best_observed)
    # #TODO: Take doe into account
    #
    # fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    #
    # ax.plot(
    #     iters,
    #     y_ei,
    #     linewidth=1.5,
    #     alpha=0.6,
    # )
    #
    # # ax.set_xlabel("number of observations (beyond initial points)")
    # ax.set_xlabel("number of observations ")
    # ax.set_ylabel("Value")
    # # ax.set_ylabel("Log10 Regret")
    # # ax.set_ylabel("Log Regret")
    # plt.show()

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