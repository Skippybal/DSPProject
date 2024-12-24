#!usr/bin/env python3

"""
"""

__author__ = "Skippybal"
__version__ = "0.1"

import json
import math
import os
import sys
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

    candidates, _ = optimize_acqf(
        acq_function=acq_func,
        bounds=func.bounds,
        q=1,
        num_restarts=num_restarts,
        raw_samples=raw_samples,  # used for intialization heuristic
        retry_on_optimization_warning=False,
        options={"batch_limit": 64, "maxiter": 300,
                 "nonnegative": False,
                 "sample_around_best": True,
                 "sample_around_best_sigma":0.1},
    )

    # observe new values
    new_x = candidates.detach()
    exact_obj = func(new_x).unsqueeze(-1)  # add output dimension
    # print("''''''''")
    # print(exact_obj)
    # print(func.evaluate_true(new_x))
    # train_obj = exact_obj + NOISE_SE * torch.randn_like(exact_obj)
    return new_x, exact_obj#train_obj


def initialize_model(train_x, train_obj, indices=None):
    # print(train_x.shape[1])
    if indices is not None:
        # print("-----------")
        # print(indices.first_order)
        # remove_neg = indices.first_order.clip(min=0)
        # print(indices)
        remove_neg = np.array(indices["ST"]).clip(min=0)
        eps = 1e-8
        inverse = 1 / (remove_neg + eps)
        # print(inverse)
        normalized = inverse / sum(inverse)# np.linalg.norm(inverse)

        # print(normalized)

        # # print(sum(normalized))
        # # print(indices.first_order / np.linalg.norm(indices.first_order))
        # # loc_thing = normalized * (np.sqrt(2) + math.log(train_x.shape[1])* 1/2) * np.sqrt(train_x.shape[1])
        # # loc_thing = np.exp(math.log(train_x.shape[1]) * 1/2 + np.sqrt(2) + (np.sqrt(3)**2)/2) #* train_x.shape[1] * normalized
        # # loc_thing = np.log(loc_thing/(np.exp(np.sqrt(3)**2/2)))
        # loc_thing = np.exp(math.log(train_x.shape[1]) * 1/2 + np.sqrt(2) + (np.sqrt(3)**2)/2) * train_x.shape[1] * normalized
        # loc_thing = np.log(loc_thing/(np.exp(np.sqrt(3)**2/2)))
        # # print(loc_thing)
        # # print(loc_thing)
        # # print(normalized)
        # # print(np.exp(np.sqrt(2) + (np.sqrt(3)**2)/2))
        # # print(np.exp(np.sqrt(2) + (np.sqrt(3)**2)/2) / np.sqrt(train_x.shape[1]))
        # # print(np.mean(loc_thing))
        # # print(loc_thing)
        # # print(np.sqrt(2) + math.log(train_x.shape[1]) * 1/2)

        # ranks = np.argsort(-indices.first_order) + 1
        # new_scales = ranks / sum(ranks)
        # normalized = new_scales
        # # print(normalized)
        # print(ranks)
        # factors = []
        # for i in range(len(indices["ST"])):
        #     if indices["ST"][i] > 0.05 and indices["ST_conf"][i] > 0.01:
        #         # factors.append(0.7)
        #         factors.append(1)
        #     else:
        #         # factors.append(1)
        #         factors.append(1.3)

        factors = []
        for i in range(len(indices["S1"])):
            if indices["S1"][i] > 0.05 and indices["S1_conf"][i] > 0.01:
                # factors.append(0.7)
                factors.append(0.8)
            else:
                # factors.append(1)
                factors.append(1)

        # print(indices["S1"])
        # # # print(indices["S1_conf"])
        # print(np.argsort(-np.array(indices["S1"])))
        # # # print(max(remove_neg)/ (min(remove_neg) + 1e-4))


        # loc_thing = np.exp(math.log(train_x.shape[1]) * 1/2 + np.sqrt(2) + (np.sqrt(3)**2)/2) * np.array(factors) #* train_x.shape[1] * normalized
        # loc_thing = np.log(loc_thing/(np.exp(np.sqrt(3)**2/2)))

        gamma = 0.2

        n_points = math.ceil( len(indices["ST"])*gamma)
        # print("-----------")
        # print(np.sqrt(2) + math.log(train_x.shape[1]) * 1/2)

        scaled_prior = np.sqrt(2) + math.log(n_points) * 1/2
        total = np.exp(math.log(train_x.shape[1]) * 1/2 + np.sqrt(2) + (np.sqrt(3)**2)/2) * train_x.shape[1]
        # print(total)
        n_bad = len(indices["ST"]) - n_points
        devided = (total - np.exp(math.log(n_points) * 1/2 + np.sqrt(2) + (np.sqrt(3)**2)/2) * n_points) / n_bad
        # print()
        # print(devided)
        bad_prior = np.log(devided/(np.exp(np.sqrt(3)**2/2)))
        # print(bad_prior)
        # print(scaled_prior)

        loc_thing = []

        ranks = np.argsort(-np.array(indices["ST"]))
        for i in range(len(indices["ST"])):
            if ranks[i] < n_points:
                loc_thing.append(scaled_prior)
            else:
                loc_thing.append(bad_prior)


        # print(loc_thing)

        # print(torch.tensor(loc_thing).unsqueeze(0).shape)

        covar_mod = RBFKernel(ard_num_dims=train_x.shape[1],
                              lengthscale_prior=LogNormalPrior(loc=torch.tensor(loc_thing).unsqueeze(0), scale=np.sqrt(3)),
                              lengthscale_constraint=GreaterThan(1e-4))
    else:
        covar_mod = RBFKernel(ard_num_dims=train_x.shape[1],
                              lengthscale_prior=LogNormalPrior(loc=np.sqrt(2) + math.log(train_x.shape[1]) * 1/2, scale=np.sqrt(3)),
                              lengthscale_constraint=GreaterThan(1e-4))

    # define the model for objective
    model = SingleTaskGP(
        train_X=train_x,
        train_Y=train_obj,
        # train_Yvar=train_yvar.expand_as(train_obj),
        # likelihood=...,
        covar_module=covar_mod,
        input_transform=Normalize(train_x.shape[1], [0,1]),
        outcome_transform=Standardize(m=1)
    ).to(train_x)
    mll = ExactMarginalLogLikelihood(model.likelihood, model)
    return mll, model

# def get_cca(model, train_x, train_y):
#     # cca = CCA(n_components=1)
#     # cca.fit(train_x, train_y)
#     #
#     # # X_c, Y_c = cca.transform(train_x, train_y)
#     #
#     # # Score the CCA model
#     # score = cca.score(train_x, train_y)
#     # print(cca.coef_)
#     # # print(np.diag(np.corrcoef(cca._x_scores, cca._y_scores, rowvar=False)[:1, 1:]))
#     # print(cca.x_loadings_)
#     # # Print the score
#     # print(score)
#
#     c_values = [0.9, 0.99]
#
#     gammas = [1e-1, 1e-2]
#     param_grid_rbf = {
#         "kernel": ["rbf"],
#         "gamma": [gammas, gammas],
#         "c": [c_values, c_values],
#     }
#
#     # Tuning hyperparameters using GridSearchCV for the Gaussian/RBF kernel.
#     kernel_rbf = (
#         GridSearchCV(
#             KCCA(latent_dimensions=1),
#             param_grid=param_grid_rbf,
#             cv=3,
#             verbose=True,
#         )
#         .fit([train_x, train_y])
#         .best_estimator_
#     )
#
#     # print(kernel_rbf)
#     correlation = kernel_rbf.score([train_x, train_y])
#     # print(correlation)
#     # print(kernel_rbf.coef0)

# def get_shaps(X, y):
#     # import interpret.glassbox
#     # import shap
#     # import pandas as pd
#     X = pd.DataFrame(X.numpy())
#     y = pd.DataFrame(y.numpy())
#     X100 = shap.utils.sample(X, 100)
#     # print(X100)
#     model_ebm = interpret.glassbox.ExplainableBoostingRegressor(interactions=0)
#     model_ebm.fit(X, y)
#     explainer_ebm = shap.Explainer(model_ebm.predict, X100)
#     shap_values_ebm = explainer_ebm(X)
#     # print(shap_values_ebm)
#     shap.plots.beeswarm(shap_values_ebm)
#     # shap.plots.bar(shap_values_ebm)
#     shap.plots.beeswarm(shap_values_ebm.abs, color="shap_red")
#     # shap.plots.bar(shap_values_ebm)
#     # shap.plots.bar(shap_values_ebm.abs.max(0))


def get_sobol_2(model, train_x, train_y):
    problem = {
        'num_vars': train_x.shape[1],
        'names': [f"x{i}" for i in range(train_x.shape[1])],
        'bounds': [[0,1] for _ in range(train_x.shape[1]) ]
    }
    # print(problem)
    # Generate samples
    param_values = sample(problem, 1024)
    def custom_func(inputs):
        # print(inputs.shape)
        # print(inputs)
        # tensor = torch.tensor(inputs)
        # print(tensor.shape)
        preds = model(torch.tensor(inputs))
        # print(preds.mean.unsqueeze(0).shape)
        return preds.mean.detach().numpy()

    # param_values = saltelli.sample(problem, 1024)
    param_values = sobol.sample(problem, 1024)
    # print(param_values)

    # Y = np.zeros([param_values.shape[0]])
    #
    # for i, X in enumerate(param_values):
    #     f = custom_func(X)
    #     print(f)
    #     Y[i] = f

    # Run model (example)
    Y = custom_func(param_values)
    # print(Y.shape)
    #
    # Perform analysis
    Si = analyze(problem, Y, print_to_console=False)

    # Print the first-order sensitivity indices
    # print(Si['S1'])
    return Si

def get_sobol(model, train_x, train_y):
    # mll, model = initialize_model(train_x, train_y)
    # fit_gpytorch_mll(mll)

    def custom_func(inputs):
        # print(inputs.shape)
        # tensor = torch.tensor(inputs)
        # print(tensor.shape)
        preds = model(torch.tensor(inputs.T))
        # print(preds.mean.unsqueeze(0).shape)
        return preds.mean.unsqueeze(0).detach().numpy()

    indices = sobol_indices(

        func=custom_func, n=2048,

        dists=[
            uniform(loc=0, scale=1),
        ] * train_x.shape[1],
        random_state=42
    )
    # print(indices.first_order)
    # print(indices.first_order)
    # remove_neg = indices.first_order.clip(min=0)
    # print(remove_neg)
    # print(sum(indices.first_order))
    # print(sum(remove_neg))
    # print(indices.first_order / np.linalg.norm(indices.first_order))
    # print("ffffffffffffff")
    # print(indices)
    return indices

def create_problem(fid: int, dims, instance):
    problem = get_problem(fid, dimension=dims, instance=instance, problem_class=ProblemClass.BBOB)
    l = logger.Analyzer(
        root="data",
        folder_name="run",
        algorithm_name="DSP",
        algorithm_info=""
    )
    problem.attach_logger(l)
    return problem, l


def main():
    torch.manual_seed(0)

    # func_org = Hartmann(dim=6, negate=True)
    # # func_tup = (Embedded, dict(function=Hartmann(dim=6), noise_std=0.01, negate=True, dim=25))
    # func_tup = (Embedded, dict(function=Hartmann(dim=6), noise_std=0.01, negate=True, dim=1000))
    # # func_tup = (Embedded, dict(function=Hartmann(dim=6), noise_std=0.01, negate=True, dim=100))
    # # func_tup = (BenchSuiteFunction, dict(negate=True, task_id='mopta'))
    # # func_tup = (MujocoFunction, dict(negate=True, bounds=[[-1,1] * 888] , container='mujoco', task_id='ant'))
    # func = func_tup[0](**func_tup[1])

    func, _logger = create_problem(21, dims=40, instance=1)
    minimize = True
    # print(func.bounds)

    NOISE_SE = 0.05
    train_yvar = torch.tensor(NOISE_SE ** 2, device=device, dtype=dtype)

    # bounds = torch.tensor([[0.0] * 6, [1.0] * 6], device=device, dtype=dtype)
    # print(bounds)

    n = 1 #20#10#14
    # generate initial training data
    print(func.bounds)
    # print(torch.)
    print(Hartmann(dim=6, negate=True).bounds)
    train_x = draw_sobol_samples(
        bounds=func.bounds, n=n, q=1, seed=torch.randint(0, 10000, (1,)).item()
    ).squeeze(1)
    exact_obj = func(train_x).unsqueeze(-1)  # add output dimension

    best_observed_value = exact_obj.max().item()
    train_obj = exact_obj #+ NOISE_SE * torch.randn_like(exact_obj)
    # print(best_observed_value)
    # print(train_obj)
    # print(train_x.shape)
    # print(train_x.shape[1])

    best_observed = [train_obj[0].item()]
    model = None

    # for i in range(1, train_obj.shape[0]):
    #     if train_obj[i].item() > best_observed[-1]:
    #         best_observed.append(train_obj[i].item())
    #     else:
    #         best_observed.append(best_observed[-1])

    for i in range(1, train_obj.shape[0]):
        if train_obj[i].item() < best_observed[-1]:
            best_observed.append(train_obj[i].item())
        else:
            best_observed.append(best_observed[-1])

    # best_observed = [best_observed_value]
    if minimize:
        train_obj = -train_obj

    mll, model = initialize_model(train_x, train_obj)

    # print(train_x[0])
    # true_vals = func.evaluate_true(train_x)
    # true_vals = [func.evaluate_true(conf) for conf in train_x]
    # print(true_vals)
    # print(exact_obj)

    N_BATCH = 1#90#40

    for iteration in range(1, N_BATCH + 1):
        # fit the models
        fit_gpytorch_mll(mll)
        ei = qLogNoisyExpectedImprovement(model=model, X_baseline=train_x)

        # optimize and get new observation
        new_x, new_obj = optimize_acqf_and_get_observation(func, ei)

        # update training points
        train_x = torch.cat([train_x, new_x])
        if minimize:
            train_obj = torch.cat([train_obj, -new_obj])
        else:
            train_obj = torch.cat([train_obj, new_obj])
        # if minimize is True:
        #     train_obj = -train_obj

        # update progress
        # TODO: This only works if we assume no noise
        # best_value = func(train_x).max().item()
        # TODO: Ask how to handle this
        if new_obj < best_observed[-1]:
            best_observed.append(new_obj.item())
            # print(new_obj)
            print(iteration + n)
            # print(np.log10(func_org.optimal_value - new_obj.item()))
            print(new_obj.item())
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

        print(".", end="")
    # print(best_observed_value)
    # print(best_observed)
    # GLOBAL_MAXIMUM = func_org.optimal_value #+ 0.1 # Add possible noise
    # print(GLOBAL_MAXIMUM)
    GLOBAL_MAXIMUM = func.optimum

    # iters = np.arange(N_BATCH + 1)
    iters = np.arange(start=1, stop=N_BATCH + n+1)
    # y_ei = np.log10(GLOBAL_MAXIMUM - np.asarray(best_observed))
    y_ei = np.log10(GLOBAL_MAXIMUM - np.asarray(best_observed))
    # y_ei = np.log(GLOBAL_MAXIMUM - np.asarray(best_observed))
    # y_ei = np.log10(GLOBAL_MAXIMUM - np.asarray(best_observed))
    # y_ei = np.asarray(best_observed)
    #TODO: Take doe into account

    fig, ax = plt.subplots(1, 1, figsize=(8, 6))

    ax.plot(
        iters,
        y_ei,
        linewidth=1.5,
        alpha=0.6,
    )

    # ax.set_xlabel("number of observations (beyond initial points)")
    ax.set_xlabel("number of observations (beyond initial points)")
    # ax.set_ylabel("Value")
    ax.set_ylabel("Log10 Regret")
    # ax.set_ylabel("Log Regret")
    plt.show()

    return 0


if __name__ == '__main__':
    exitcode = main()
    sys.exit(exitcode)