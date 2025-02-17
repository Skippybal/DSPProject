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
# from cca_zoo.model_selection import GridSearchCV
# from cca_zoo.nonparametric import KCCA
#
# from SALib.analyze.sobol import analyze
# from SALib.sample.sobol import sample
# from SALib.sample import saltelli, sobol

# from Extra.bench import BenchSuiteFunction
from Extra.mujo import MujocoFunction
from Extra.synth import Embedded
from scipy.stats import sobol_indices, uniform

# import interpret.glassbox
# import shap
# import pandas as pd

from ioh import get_problem, logger, ProblemClass

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dtype = torch.double
SMOKE_TEST = os.environ.get("SMOKE_TEST")
print(device)


# def optimize_acqf_and_get_observation(func, acq_func, num_restarts=20, raw_samples=512, NOISE_SE=0.05):
def optimize_acqf_and_get_observation(func, acq_func, args,num_restarts=20, raw_samples=512):
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
        q=args.pp, #1,#5,
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


    end = time.time()

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
        root="results/DSP",
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
    parser.add_argument('--pp', type=int, default=1)

    return parser.parse_args()


def main_loop(args, func, minimize=True):
    # torch.manual_seed(0)

    n = args.doe #20 #20#10#14


    train_x = draw_sobol_samples(
        bounds=torch.tensor(np.array([func.bounds.lb, func.bounds.ub])), n=n, q=1, seed=torch.randint(0, 10000, (1,)).item()
    ).squeeze(1)
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
    if minimize:
        train_obj = [-i for i in train_obj]

    train_obj = torch.tensor(np.array(train_obj).reshape(-1,1)) #.to(device)

    mll, model = initialize_model(train_x, train_obj)
    N_BATCH = args.total-args.doe#90#40

    # for iteration in range(1, ((N_BATCH + 1)//args.pp)+1):
    # for iteration in range(1, N_BATCH + 1 + 1):
    for iteration in range(1, math.ceil(N_BATCH / args.pp) + 1):
        # fit the models
        start = time.time()
        fit_gpytorch_mll(mll, approx_mll=True)
        end = time.time()

        ei = qLogNoisyExpectedImprovement(model=model, X_baseline=train_x)

        # optimize and get new observation
        new_x, new_obj = optimize_acqf_and_get_observation(func, ei, args)

        train_x = torch.cat([train_x, new_x])
        if minimize:
            train_obj = torch.cat([train_obj, torch.tensor(-np.array(new_obj).reshape(-1,1))]) #.to(device)
        else:
            train_obj = torch.cat([train_obj, torch.tensor(np.array(new_obj).reshape(-1,1))])#.to(device)



        if new_obj[0] < best_observed[-1]:
            best_observed.append(new_obj[0])

        else:
            best_observed.append(best_observed[-1])

        # best_value = func(train_x).min().item()
        # best_observed.append(best_value)
        # if i > 1:
        # indices = get_sobol(model, train_x, train_obj)

        # indices = get_sobol_2(model, train_x, train_obj)
        indices = None


        mll, model = initialize_model(train_x, train_obj, indices)

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