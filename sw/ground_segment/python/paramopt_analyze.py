#!/usr/bin/env python

# plot modes / FIXME
# - plot timeseries
# - plot mse chronologically / sorted
# - plot mse and params chronologically / sorted
# - number of runs and interruptions
# - mutual information between params and MSE
# - superimposed ensemble of timeseries

from __future__ import print_function

import argparse, sys, os, time
import numpy as np
import matplotlib.pylab as pl
from matplotlib import gridspec
from matplotlib import cm
import pandas as pd
import tables as tb

from paramopt_data import PPRZ_Opt_Attitude

modes = {
    "plot_timeseries": 0,
    "plot_mse": 1,
    "plot_mse_pid": 2,
    "plot_best_to_date": 3
    }

def load_table(args):
    tblfilename = args.logfiles[0][0]
    print("tblfilename", tblfilename)
    h5file = tb.open_file(tblfilename, mode = "a")
    return h5file.root.v2.evaluations
    
def clean_logdata(logdata):
    rows, cols = logdata.shape
    for j in range(cols):
        for i in range(1, rows):
            if logdata[i,j] == 0.:
                logdata[i,j] = logdata[i-1,j]
    return logdata
    
def plot_timeseries(args):
    print("plot_timeseries", args)

    table = load_table(args)
    print("table", table, table.nrows)
    mses = np.zeros((table.nrows, 2))
    idx = 0
    for x in table.iterrows():
        print("x", x["timeseries"].shape)
        data = clean_logdata(x["timeseries"])

        # stats
        print("raw mse = %f" % (x["mse"]))
        c1 = np.mean(np.square(data[:,1] - data[:,2]))
        c2 = np.mean(np.square(data[:,4] - data[:,5]))
        print("new mse = %f" % (c1 + c2))
        mses[idx,0] = x["mse"]
        mses[idx,1] = c1+c2
        
        pl.subplot(211)
        pl.plot(data[:,0:3])
        pl.legend(("sp_phi", "ref_phi", "est_phi"))
        pl.subplot(212)
        pl.plot(data[:,3:6])
        pl.legend(("sp_theta", "ref_theta", "est_theta"))
        pl.show()
        idx += 1
    print("mse", np.mean(mses, 0), np.var(mses, 0))

def plot_mse(args):
    print("plot_timeseries", args)
    table = load_table(args)
    mse  = [x["mse"] for x in table.iterrows()]

    pl.plot(mse)
    pl.show()

def plot_mse_pid(args):
    table = load_table(args)

    data = [[x["pgain_phi"],
             x["igain_phi"],
             x["dgain_p"],
             x["ddgain_p"],
             x["pgain_theta"],
             x["igain_theta"],
             x["dgain_q"],
             x["ddgain_q"],
             x["mse"]
             ] for x in table.iterrows()]
    data = np.asarray(data)

    mse_sorted_idx = np.argsort(data[:,8])
    # print("best five", mse_sorted[:5])
    print("best five", data[mse_sorted_idx[:5],:])
        
    pl.subplot(511)
    pl.plot(data[:,8])
    pl.subplot(512)
    pl.plot(data[:,[0,4]])
    pl.subplot(513)
    pl.plot(data[:,[1,5]])
    pl.subplot(514)
    pl.plot(data[:,[2,6]])
    pl.subplot(515)
    pl.plot(data[:,[3,7]])
    pl.show()

def plot_best_to_date(args):
    # print("implement me")
    table = load_table(args)
    data = [[x["pgain_phi"],
             x["igain_phi"],
             x["dgain_p"],
             x["ddgain_p"],
             x["pgain_theta"],
             x["igain_theta"],
             x["dgain_q"],
             x["ddgain_q"],
             x["mse"]
             ] for x in table.iterrows()]
    data = np.asarray(data)

    best_to_date = np.zeros((data.shape[0], 1))


    best_to_date_start = 0
    best_to_date[0:best_to_date_start,0] = np.mean(data[0:best_to_date_start,8])
    for i in range(best_to_date_start, data.shape[0]):
        # print(i)
        best_to_date[i,0] = np.min(data[best_to_date_start:i+1,8])

    print("first pid", data[0,:])
        
    pl.plot(best_to_date)
    pl.plot(data[:,8], lw=0.2)
    pl.text(10, 20, "min = %f" % (np.min(data[:,8])))
    pl.gca().set_yscale("log")
    pl.show()
        
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-l', "--logfiles", action='append', dest='logfiles',
                        default=[], nargs = "+",
                        help='Add logfiles for analysis')
    parser.add_argument("-m", "--mode", dest="mode", help="Mode, one of " + ", ".join(modes.keys()), default="plot_timeseries")

    args = parser.parse_args()
    if len(args.logfiles) < 1:
        print("need to pass at least one logfile")
        sys.exit(1)

    if args.mode == "plot_timeseries":
        plot_timeseries(args)
    elif args.mode == "plot_mse":
        plot_mse(args)
    elif args.mode == "plot_mse_pid":
        plot_mse_pid(args)
    elif args.mode == "plot_best_to_date":
        plot_best_to_date(args)
    else:
        print("unknown mode '%s'" % (args.mode))
        
