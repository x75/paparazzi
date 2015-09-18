#!/usr/bin/env python
"""Test hp_gpsmbo on a simple function with two arguments"""

from __future__ import print_function

import argparse
from functools import partial
import numpy as np
import pylab as pl

import hyperopt
from hyperopt import hp
from hyperopt import Trials, fmin, rand, tpe
# from hyperopt import hp

import hp_gpsmbo.hpsuggest
from hp_gpsmbo import suggest_algos

from mpl_toolkits.mplot3d import Axes3D

def myfunc(x, y):
    # R = np.sin(0.8 * x) + np.cos(0.33 * y)
    R = (0.8 * (x + 1.7) ** 2) + (0.33 * (y + 2.3) ** 2)
    return R
    

def objective(params):
    print("################################################################################")
    print("params", params)
    return myfunc(params["x"], params["y"])

if __name__ == "__main__":
    # print("main")
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--mode", default="rand")
    parser.add_argument("-p", "--plot", action="store_true")

    args = parser.parse_args()
    
    t1 = np.linspace(-1., 1.5 * np.pi, 101)
    t2 = np.linspace(-1., 3.7 * np.pi, 101)
    T = np.meshgrid(t1, t2)
    R = myfunc(T[0], T[1])
    R_argmin = np.argmin(R)
    print("argmin", R_argmin, R[R_argmin], R.shape)
    print("min", np.min(R), R.shape)

    if args.plot:
        fig = pl.figure()
        ax = fig.gca(projection="3d")
        surf = ax.plot_surface(T[0], T[1], R)
        pl.show()

    space = {
        "x": hp.uniform("x", -1., 1.5 * np.pi),
        "y": hp.uniform("y", -1., 3.7 * np.pi)
        }

    # print(dir(hp_gpsmbo.hpsuggest))
    if args.mode == "gp_ucb":
        # suggest_ucb = 
        suggest = partial(suggest_algos.ucb, stop_at=1.),
        # print("suggest", suggest, type(suggest))
        suggest = suggest[0]
    elif args.mode == "gp_ei":
        suggest = partial(suggest_algos.ei, stop_at=1.),
        suggest = suggest[0]
    elif args.mode == "tpe":
        suggest = tpe.suggest
    else:
        suggest = rand.suggest

    # final suggest
    # suggest = partial(suggest_ei, stop_at=5.),
    
    trials = Trials()

    # print("types", type(objective), type(trials), type(suggest), suggest, type(np.random.RandomState(1)))
    
    best = fmin(fn=objective,
                space=space,
                trials=trials,
                # algo=partial(suggest, stop_at=-1.2),
                algo=suggest,
                rstate=np.random.RandomState(1),
                max_evals=1000)

    print("best", best)
