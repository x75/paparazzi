#!/usr/bin/env python
"""Test hp_gpsmbo on a simple function with two arguments"""

from __future__ import print_function

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
    print("main")
    
    t1 = np.linspace(0, 1.5 * np.pi, 101)
    t2 = np.linspace(0, 3.7 * np.pi, 101)
    T = np.meshgrid(t1, t2)
    # fig = pl.figure()
    # ax = fig.gca(projection="3d")
    R = myfunc(T[0], T[1])
    print("argmin", np.argmin(R), R.shape)
    print("min", np.min(R), R.shape)
    # surf = ax.plot_surface(T[0], T[1], R)
    # pl.show()

    
    space = {
        "x": hp.uniform("x", 0, 1.5 * np.pi),
        "y": hp.uniform("y", 0, 3.7 * np.pi)
        }

    # print(dir(hp_gpsmbo.hpsuggest))
    suggest_ucb = suggest_algos.ucb
    suggest_ei = suggest_algos.ei
    suggest_tpe = tpe.suggest
    suggest_rand = rand.suggest
    suggest = partial(suggest_ei, stop_at=0.2),
    trials = Trials()
    best = fmin(fn=objective,
                space=space,
                trials=trials,
                # algo=partial(suggest, stop_at=-1.2),
                algo=
                rstate=np.random.RandomState(1),
                max_evals=100)

    print("best", best)
