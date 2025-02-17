import argparse
import math

import GPy
# import matlab.engine
import numpy as np
import torch
from botorch import fit_gpytorch_mll
from botorch.acquisition import qLogNoisyExpectedImprovement
from botorch.models import SingleTaskGP
from botorch.models.transforms import Normalize, Standardize
from botorch.optim import optimize_acqf
from gpytorch import ExactMarginalLogLikelihood
# from gpytorch.kernels import RBFKernel
from gpytorch.priors import LogNormalPrior
from ioh import get_problem, logger, ProblemClass
from pyDOE import lhs
# from sympy import GreaterThan
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
from gpytorch.constraints.constraints import GreaterThan
import functions
from REMBO import EI
import timeit
import time

np.random.seed(42)

def dim_sampling(low_dim, X, bx_size):
    if len(X.shape)==1:
        X=X.reshape((1, X.shape[0]))
    n=X.shape[0]
    high_dim=X.shape[1]
    low_obs=np.zeros((n,low_dim))
    high_to_low=np.zeros(high_dim,dtype=int)
    sign=np.random.choice([-1,1],high_dim)
    for i in range(high_dim):
        high_to_low[i]=np.random.choice(range(low_dim))
        low_obs[:,high_to_low[i]]=X[:,i]*sign[i]+ low_obs[:,high_to_low[i]]

    for i in range(n):
        for j in range(low_dim):
            if low_obs[i][j] > bx_size: low_obs[i][j] = bx_size
            elif low_obs[i][j] < -bx_size: low_obs[i][j] = -bx_size
    return low_obs, high_to_low, sign

def back_projection(low_obs, high_to_low, sign, bx_size):
    if len(low_obs.shape)==1:
        low_obs=low_obs.reshape((1, low_obs.shape[0]))
    n=low_obs.shape[0]
    high_dim=high_to_low.shape[0]
    low_dim=low_obs.shape[1]
    high_obs=np.zeros((n,high_dim))
    scale=1
    for i in range(high_dim):
        high_obs[:,i]=sign[i]*low_obs[:,high_to_low[i]]*scale
    for i in range(n):
        for j in range(high_dim):
            if high_obs[i][j] > bx_size: high_obs[i][j] = bx_size
            elif high_obs[i][j] < -bx_size: high_obs[i][j] = -bx_size
    return high_obs


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--folder_name', type=str, default='run')
    parser.add_argument('--algo_name', type=str, default='HeSBO')
    parser.add_argument('--dims', type=int, default=10)
    parser.add_argument('--func', type=int, default=21)
    parser.add_argument('--n_trails', type=int, default=5)
    parser.add_argument('--doe', type=int, default=20)
    parser.add_argument('--total', type=int, default=150)
    parser.add_argument('--instance', type=int, default=1)
    parser.add_argument('--minimize', type=bool, default=True)
    parser.add_argument('--low_dims', type=int, default=3)

    return parser.parse_args()


def create_problem(fid: int, args):
    problem = get_problem(fid, dimension=args.dims, instance=args.instance, problem_class=ProblemClass.BBOB)
    l = logger.Analyzer(
        root="results/HeSBO_DSP",
        folder_name=args.folder_name,
        algorithm_name=args.algo_name,
        algorithm_info=""
    )
    problem.attach_logger(l)
    return problem, l

def RunMain(func, low_dim=2, high_dim=25, initial_n=20, total_itr=100, func_type='Branin',
            s=None, active_var=None, ARD=False, variance=1., length_scale=None, box_size=None,
            high_to_low=None, sign=None, hyper_opt_interval=20, noise_var=0, DSP=False):
    """

    :param high_dim: the dimension of high dimensional search space
    :param low_dim: The effective dimension of the algorithm.
    :param initial_n: the number of initial points
    :param total_itr: the number of iterations of algorithm. The total
        number of test function evaluations is initial_n + total_itr
    :param func_type: the name of test function
    :param s: initial points
    :param active_var: a vector with the size of greater or equal to
        the number of active variables of test function. The values of
        vector are integers less than high_dim value.
    :param ARD: if TRUE, kernel is isomorphic
    :param variance: signal variance of the kernel
    :param length_scale: length scale values of the kernel
    :param box_size: this variable indicates the search space [-box_size, box_size]^d
    :param high_to_low: a vector with D elements. each element can have a value from {0,..,d-1}
    :param sign: a vector with D elements. each element is either +1 or -1.
    :param hyper_opt_interval: the number of iterations between two consecutive
        hyper parameters optimizations
    :param noise_var: noise variance of the test functions
    :return: a tuple of best values of each iteration, all observed points, and
        corresponding test function values of observed points
    """

    if active_var is None:
        active_var= np.arange(high_dim)
    if box_size is None:
        box_size=1
    if high_to_low is None:
        high_to_low=np.random.choice(range(low_dim), high_dim)
    if sign is None:
        sign = np.random.choice([-1, 1], high_dim)

    #Specifying the type of objective function
    # if func_type=='Branin':
    #     test_func = functions.Branin(active_var, noise_var=noise_var)
    # elif func_type=='Rosenbrock':
    #     test_func = functions.Rosenbrock(active_var, noise_var=noise_var)
    # elif func_type=='Hartmann6':
    #     test_func = functions.Hartmann6(active_var, noise_var=noise_var)
    # elif func_type == 'StybTang':
    #     test_func = functions.StybTang(active_var, noise_var=noise_var)
    # else:
    #     TypeError('The input for func_type variable is invalid, which is', func_type)
    #     return
    # test_func = func
    def prox_func(x):
        return [-i for i in func(x)]
    test_func = prox_func

    best_results = np.zeros([1, total_itr + initial_n])
    elapsed=np.zeros([1, total_itr + initial_n])

    # Creating the initial points. The shape of s is nxD
    if s is None:
        s=lhs(low_dim, initial_n) * 2 * box_size - box_size

    f_s = np.array(test_func(back_projection(s,high_to_low,sign,box_size))).reshape(-1,1)


    # Main loop
    for i in range(total_itr):

        mll, model = initialize_model(s, f_s)
        fit_gpytorch_mll(mll, approx_mll=True)


        start = timeit.default_timer()


        # Maximizing acquisition function
        D = lhs(low_dim, 2000) * 2 * box_size - box_size

        v = model(torch.tensor(D))

        f_mean = v.mean
        f_var = v.variance

        stdada = Standardize(m=1)
        # print(stdada(train_obj))
        max_f = max(stdada(torch.tensor(f_s))[0])

        ei_d = EI(len(D), max_f, f_mean.detach().numpy(), f_var.detach().numpy())
        index = np.argmax(ei_d)

        # Adding the new point to our sample
        s = np.append(s, [D[index]], axis=0)
        new_high_point=back_projection(D[index],high_to_low,sign,box_size)

        f_s = np.append(f_s, np.array(test_func(new_high_point)).reshape(-1,1), axis=0)


        stop = timeit.default_timer()
        elapsed[0, i + initial_n]=stop-start

    return elapsed

def optimize_acqf_and_get_observation(func, acq_func, low_dims, num_restarts=20, raw_samples=512):
    """Optimizes the acquisition function, and returns a new candidate and a noisy observation."""

    start = time.time()

    candidates, _ = optimize_acqf(
        acq_function=acq_func,
        bounds=torch.tensor(np.array([func.bounds.lb[:low_dims], func.bounds.ub[:low_dims]])), #.to(device),
        q=1,#5,
        num_restarts=5,
        raw_samples=256,  # used for intialization heuristic
        retry_on_optimization_warning=False,
        options={"batch_limit": 64, "maxiter": 100,
                 "nonnegative": False,
                 "sample_around_best": True,
                 "sample_around_best_sigma": 0.1},
    )

    # observe new values
    new_x = candidates.detach()
    # exact_obj = func(new_x.cpu().numpy())

    end = time.time()
    # print(end - start)

    return new_x

def initialize_model(train_x, train_obj):
    train_x = torch.tensor(train_x)
    train_obj = torch.tensor(train_obj).reshape(-1,1)

    covar_mod = RBFKernel(ard_num_dims=train_x.shape[1],
                          lengthscale_prior=LogNormalPrior(loc=np.sqrt(2) + math.log(train_x.shape[1]) * 1/2, scale=np.sqrt(3)),
                          lengthscale_constraint=GreaterThan(1e-4))

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

if __name__=='__main__':

    args = parse_args()
    func, _logger = create_problem(args.func, args=args)
    # minimize = True
    for i in range(args.n_trails):
        print(f"\nStarting trail {i}")

        time_el = RunMain(func, low_dim=args.low_dims, high_dim=args.dims, initial_n=args.doe, total_itr=args.total, ARD=False,
                                                 noise_var=0, box_size=func.bounds.ub[0],
                       DSP=True)


        func.reset()
    _logger.close()
