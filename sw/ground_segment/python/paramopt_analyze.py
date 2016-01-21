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

# jpype for jidt
from jpype import *

from paramopt_data import PPRZ_Opt_Attitude

modes = {
    "plot_timeseries": 0,
    "plot_mse": 1,
    "plot_mse_pid": 2,
    "plot_best_to_date": 3
    }

files = {"anneal": ["paramopt_ppz_v2_anneal_1.h5",
                    "paramopt_ppz_v2_anneal_2.h5",
                    "paramopt_ppz_v2_anneal_3.h5"
                    ],
        # "gp_ei": [
        #     "paramopt_ppz_v2_gp_ei_1.h5",
        #     "paramopt_ppz_v2_gp_ei_2.h5",
        #           ],
        "gp_ucb": ["paramopt_ppz_v2_gp_ucb_1.h5",
                   "paramopt_ppz_v2_gp_ucb_2.h5",
                   "paramopt_ppz_v2_gp_ucb_3.h5",
                   "paramopt_ppz_v2_gp_ucb_4.h5",
                   ],
        "rand": ["paramopt_ppz_v2_rand_1.h5",
                 "paramopt_ppz_v2_rand_2.h5",
                 "paramopt_ppz_v2_rand_3.h5"
                 ],
        "tpe": ["paramopt_ppz_v2_tpe_1.h5",
                "paramopt_ppz_v2_tpe_2.h5",
                "paramopt_ppz_v2_tpe_3.h5"
                ]
        }

params_attitude = ["pgain_phi",
             "igain_phi",
             "dgain_p",
             "ddgain_p",
             "pgain_theta",
             "igain_theta",
             "dgain_q",
             "ddgain_q"]
            
# helper functions
# def load_table(args):
def load_table(logfile):
    # tblfilename = args.logfiles[0][0]
    tblfilename = logfile
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

def comp_best_to_date(data):
    """compute running minimum"""
    # print("comp_best_to_date data.shape", data.shape)
    best_to_date = np.zeros((data.shape[0], 1))

    best_to_date_start = 0
    if best_to_date_start > 0:
        best_to_date[0:best_to_date_start,0] = np.mean(data[0:best_to_date_start,])
    for i in range(best_to_date_start, data.shape[0]):
        # print(i)
        best_to_date[i,0] = np.min(data[best_to_date_start:i+1,])
    return best_to_date

# plot / analysis modes    
def plot_timeseries(args):
    print("plot_timeseries", args)

    # table = load_table(args)
    table = load_table(args.logfiles[0][0])
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
    # table = load_table(args)
    table = load_table(args.logfiles[0][0])
    mse  = [x["mse"] for x in table.iterrows()]

    pl.plot(mse)
    pl.show()

def plot_mse_pid(args):
    # table = load_table(args)
    table = load_table(args.logfiles[0][0])

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

    sl = slice(None)
    sl = mse_sorted_idx.flatten().tolist()
    print("type", type(sl), sl)
            
    pl.subplot(511)
    pl.plot(data[sl,8], "o")
    pl.ylabel("MSE")
    pl.gca().set_yscale("log")
    pl.subplot(512)
    # pl.plot(data[sl,[0,4]], "o")
    pl.plot(data[sl,0], "o")
    pl.plot(data[sl,4], "o")
    pl.ylabel("P")
    pl.subplot(513)
    # pl.plot(data[sl,[1,5]], "o")
    pl.plot(data[sl,1], "o")
    pl.plot(data[sl,5], "o")
    pl.ylabel("I")
    pl.subplot(514)
    # pl.plot(data[sl,[2,6]], "o")
    pl.plot(data[sl,2], "o")
    pl.plot(data[sl,6], "o")
    pl.ylabel("D")
    pl.subplot(515)
    # pl.plot(data[sl,[3,7]], "o")
    pl.plot(data[sl,3], "o")
    pl.plot(data[sl,7], "o")
    pl.ylabel("DD")
    pl.show()
    
def plot_best_to_date(args):
    # print("implement me")
    # table = load_table(args)
    table = load_table(args.logfiles[0][0])
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

    best_to_date = comp_best_to_date(data[:,8])

    print("first pid", data[0,:])
        
    pl.plot(best_to_date)
    pl.plot(data[:,8], lw=0.2)
    pl.text(10, 20, "min = %f" % (np.min(data[:,8])))
    pl.gca().set_yscale("log")
    pl.show()

def plot_best_to_date_avg_all(args):
    # print("files", files)
    mse_data = {}
    # loop over algorithm types
    for k,v in files.items():
        # print(k,v)
        mse_runs = np.zeros((60, len(v)))
        # loop over runs
        for i,run in enumerate(v):
            # print("run", run)
            table = load_table(run)
            mses = table.col("mse")
            # print("k", k, type())
            mse_runs[0:min(60, mses.shape[0]),i] = mses[0:60]
        mse_data[k] = mse_runs.copy()
    # print(mse_data)

    # loop over items in data
    for k,v in mse_data.items():
        # v is array of mse values, one column per run
        v_mean = np.mean(v, axis=1)
        # print the mean
        # print("mean", v_mean)
        # pl.plot(v, label=k, lw=0.2)
        # pl.subplot(211)
        btd_mean = np.zeros_like(v_mean)
        for i in range(v.shape[1]):
            print("num v", k, i, v.shape)
            # pl.cla()
            # pl.title("%s-%d" % (k, i))
            # pl.plot(v[:,i])
            # pl.show()
            btd_mean += comp_best_to_date(v[:,i]).reshape(btd_mean.shape)
            # pl.plot(comp_best_to_date(v[:,i]), label=k, lw=1.)
        # pl.gca().set_yscale("log")
        pl.subplot(111)
        sl = slice(0, None)
        if k == "gp_ei": # problems here with eval times
            sl = slice(0, 40)
        pl.plot((btd_mean/(i+1))[sl], "x-", label=k, lw=1.)
        # pl.plot(comp_best_to_date(v_mean), label=k, lw=1.)
        # pl.plot(v_mean, label=k, lw=1.)
    # pl.gca().set_yscale("log")
    pl.title("Average best-to-date")
    # pl.axvline(14, 0, 1.)
    pl.axvspan(0, 14, 0, 1., facecolor="k", alpha=0.1)
    pl.text(5,   6, "Warm-up")
    pl.text(25, 11, "Optim. suggestions")
    pl.xlabel("Eval #")
    pl.ylabel("Best MSE to date")
    pl.legend()
    if args.plotsave:
        # for paper
        # 0.39370079
        fig_width  = 42 * 0.39370079
        fig_height = 15 * 0.39370079
        pl.gcf().set_size_inches((fig_width, fig_height))
        # for poster
        # pl.gcf().set_size_inches((15, 10))
        pl.gcf().savefig("%s.pdf" % (sys.argv[0][:-3]), dpi=300, bbox_inches="tight")
    pl.show()
            
    # pass

def plot_variance_baseline(args):
    var_data = []
    var_data_len_tot = 0
    mses = []
    for lf in args.logfiles[0]:
        # print("lf", lf)
        # table = load_table(args)
        var_data.append(load_table(lf))
        var_data_len_tot += var_data[-1].nrows
        mses.append(var_data[-1].col("mse"))

    # var_data_raw = np.zeros((var_data_len_tot, 1))
    print("mses", mses)
    mses_flat = np.hstack(mses)
    mses_clean = mses_flat[(mses_flat < 50.)]
    mses_clean_mean = np.mean(mses_clean)
    # mses_clean = np.where(mses_flat[(mses_flat < 80.)]
    
    # pl.plot(mses_clean)
    # pl.axhline(mses_clean_mean)
    # pl.axhline(mses_clean_mean + np.std(mses_clean))
    # pl.axhline(mses_clean_mean - np.std(mses_clean))
    pl.title("Baseline histogram")
    pl.hist(mses_clean, bins=5)
    pl.xlabel("MSE")
    pl.ylabel("Count")
    if args.plotsave:
        pl.gcf().set_size_inches((10, 3))
        pl.gcf().savefig("%s.pdf" % (sys.argv[0][:-3]), dpi=300, bbox_inches="tight")
    pl.show()
    
def plot_error_ensemble(args):
    # "paramopt_ppz_v2_tpe_1_extended.h5",
    # print("files", files)
    # pass
    table = load_table(files["tpe"][1])
    rcnt = 0
    mses = table.col("mse")
    minidx = np.argmin(mses)
    maxidx = np.argmax(mses)
    selection = [minidx, 10, 20, 30]
    print("selection", selection)
    for x in table.iterrows():
        # if rcnt % 10 == 0:
        if rcnt in selection:
            print("x", x["timeseries"].shape)
            print(x)
            # pl.plot(x["timeseries"][:,0:6])
            pl.plot(np.square(x["timeseries"][:,1] - x["timeseries"][:,2]))
        rcnt += 1
    # pl.gca().set_yscale("log")
    pl.show()

def plot_mi_pid_mse_all(args):
    mis_all_a = [] # np.zeros((len(files), len(params_attitude)))
    mis_all   = {}
    pids_top_10 = []

    j = 0
    # pl.subplot(211)
    for k,v in files.items():
        # print("v", v)
        for i,v_ in enumerate(v):
            args.logfiles[0][0] = v_
            mis = plot_mi_pid_mse(args)
            pids_top_10_l = plot_hist_top_10(args)
            pids_top_10.append(pids_top_10_l)
            print("mis", k, i, mis)
            mis_all["%s_%d" % (k, i)] = mis.copy()
            mis_all_a.append(mis.copy())
            # pl.plot(mis.T, "o")
            j += 1
    # pl.subplot(212)
    # print(np.vstack(mis_all_a))
    # pl.plot(np.vstack(mis_all_a).T, "o")
    # print(mis_all)
    # pl.show()

    pl.title("Mutual information I(Param; MSE)")
    pl.boxplot(np.vstack(mis_all_a))
    pl.gca().set_xticklabels(params_attitude, fontsize=8.)
    pl.ylabel("I(param;MSE)" % (params_attitude))
    if args.plotsave:
        pl.gcf().set_size_inches((10, 4))
        pl.gcf().savefig("%s-mi-param-mse-boxplot.pdf" % (sys.argv[0][:-3]),
                         dpi=300, bbox_inches="tight")
    pl.show()

    pids_top_10_stacked = np.vstack(pids_top_10)
    print("pids top 10", pids_top_10_stacked.shape, pids_top_10_stacked)
    gs = gridspec.GridSpec(4, 2) # (5, 2)
    gs_map = [0, 2, 4, 6, 1, 3, 5, 7] # , 8]
    ylabel_map = ["pgain", "igain", "dgain", "ddgain"]
    for i in range(len(params_attitude)): # + 1)
        # print(i)
        # pl.subplot
        # idx = (i*2)%7
        # print("idx = %d" % idx)
        pl.subplot(gs[gs_map[i]])
        if i == 0:
            pl.title("Phi (roll)")
        elif i == 4:
            pl.title("Theta (pitch)")
        # pl.title("%s" % (params_attitude[i]))
        pl.hist(pids_top_10_stacked[:,i], bins=10, rwidth=0.8, color="black", alpha=0.5)
        if i < 4:
            pl.ylabel("%s" % ylabel_map[i])
    if args.plotsave:
        pl.gcf().set_size_inches((10, 6))
        pl.gcf().savefig("%s-mi-param-mse-histo.pdf" % (sys.argv[0][:-3]),
                         dpi=300, bbox_inches="tight")
    pl.show()
        

def plot_hist_top_10(args):
    tblfilename = args.logfiles[0][0]
    h5file = tb.open_file(tblfilename, mode = "a")
    table = h5file.root.v2.evaluations
    
    
    # put stuff
    if not args.sorted:
        pids = [[
            x[params_attitude[0]],
            x[params_attitude[1]],
            x[params_attitude[2]],
            x[params_attitude[3]],
            x[params_attitude[4]],
            x[params_attitude[5]],
            x[params_attitude[6]],
            x[params_attitude[7]],
            x["mse"]
             ] for x in table.iterrows()]
#        pids = [ [x["alt_p"], x["alt_i"], x["alt_d"], x["vel_p"], x["vel_i"], x["vel_d"]]
#             for x in table.iterrows() ]
        mses = [ [x["mse"]] for x in table.iterrows() ]
    else:
        print("sorted not supported here yet")
    pids = np.asarray(pids)
        
    # compare with historgram
    mse_sorted_idx = np.argsort(pids[:,8])
    # print("best five", mse_sorted[:5])

    # return the best
    best = 8
    return pids[mse_sorted_idx[:best],:]
        
   
def plot_mi_pid_mse(args):
    """compute mutual information between pid params and mse

    run like
    for f in paramopt_ppz_v2_anneal_?.h5 paramopt_ppz_v2_gp_ei_?.h5 paramopt_ppz_v2_gp_ucb_?.h5 paramopt_ppz_v2_rand_?.h5 paramopt_ppz_v2_tpe_?.h5 ; do python paramopt_analyze.py -l $f -m plot_mi_pid_mse ; done
    """
    
    tblfilename = args.logfiles[0][0]
    h5file = tb.open_file(tblfilename, mode = "a")
    table = h5file.root.v2.evaluations

    # sort rows
    if not table.cols.mse.is_indexed:
        table.cols.mse.createCSIndex()

    # put stuff
    if not args.sorted:
        pids = [[
            x[params_attitude[0]],
            x[params_attitude[1]],
            x[params_attitude[2]],
            x[params_attitude[3]],
            x[params_attitude[4]],
            x[params_attitude[5]],
            x[params_attitude[6]],
            x[params_attitude[7]],
            x["mse"]
             ] for x in table.iterrows()]
#        pids = [ [x["alt_p"], x["alt_i"], x["alt_d"], x["vel_p"], x["vel_i"], x["vel_d"]]
#             for x in table.iterrows() ]
        mses = [ [x["mse"]] for x in table.iterrows() ]
    else:
        print("sorted not supported here yet")
        
    # I think this is a bit of a hack, python users will do better on this:
    sys.path.append("../../infodynamics-dist/demos/python")
    
    jarLocation = "../../../../infodynamics-dist/infodynamics.jar"
    # Start the JVM (add the "-Xmx" option with say 1024M if you get crashes due to not enough memory space)
    if not isJVMStarted():
        print("Starting JVM")
        # startJVM(getDefaultJVMPath(), "-ea", "-Xmx8192M", "-Djava.class.path=" + jarLocation)
        startJVM(getDefaultJVMPath(), "-ea", "-Djava.class.path=" + jarLocation)
    else:
        print("Attaching JVM")
        attachThreadToJVM()

    # mutual information
    # 1. Construct the calculator:
    calcClassMI = JPackage("infodynamics.measures.continuous.kraskov").MutualInfoCalculatorMultiVariateKraskov1
    # print(calcClassMI)
    calcMI = calcClassMI()
    # 2. Set any properties to non-default values:
    # calcMI.setProperty("TIME_DIFF", "1")
    # 3. Initialise the calculator for (re-)use:
    calcMI.initialise()
    
    calcClass = JPackage("infodynamics.measures.continuous.kraskov").TransferEntropyCalculatorKraskov
    calc = calcClass()
    # 2. Set any properties to non-default values:
    calc.setProperty("k_HISTORY", "1")
    # calc.setProperty("k_TAU", "2")
    calc.setProperty("l_HISTORY", "1")
    # calc.setProperty("l_TAU", "2")
    # 3. Initialise the calculator for (re-)use:
    calc.initialise()

    pids_a = np.asarray(pids).astype(np.float64)

    # # select best ten
    # mse_sorted_idx = np.argsort(pids_a[:,8])
    # pids_a = pids_a[mse_sorted_idx[:10],:]

    # dest = np.asarray(mses).astype(np.float64).flatten()
    dest = np.asarray(pids_a[:,8]).astype(np.float64).flatten()

    # storage
    mis = np.zeros((1, len(params_attitude)))

    # print("pids_a.shape", pids_a.shape)

    for i in range(len(params_attitude)):
        source = pids_a[:,i]
        # print(source.shape, dest.shape)

        calcMI.initialise()
        calcMI.setObservations(source, dest)
        # 5. Compute the estimate:
        result = calcMI.computeAverageLocalOfObservations()
        print("MI(%s;MSE) = %.4f nats" % (params_attitude[i], result))
        mis[0,i] = result

        # calc.initialise()        
        # calc.setObservations(source, dest)
        # # 5. Compute the estimate:
        # result = calc.computeAverageLocalOfObservations()
        # print("mse: %f, TE_Kraskov (KSG)(col_0 -> col_1) = %.4f nats" % (x["mse"], result))
    return mis
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-l', "--logfiles", action='append', dest='logfiles',
                        default=[], nargs = "+",
                        help='Add logfiles for analysis')
    parser.add_argument("-m", "--mode", dest="mode", help="Mode, one of " + ", ".join(modes.keys()), default="plot_timeseries")
    parser.add_argument('-ps', "--plotsave", action='store_true', help='Save plot to pdf?')
    parser.add_argument('-s', "--sorted", dest="sorted",
                        action='store_true', help='Sort table by MSE')
    parser.add_argument('-ns', "--no-sorted", dest="sorted",
                        action='store_false', help='Sort table by MSE')

    args = parser.parse_args()
    if args.mode not in ["plot_error_ensemble", "plot_best_to_date_avg_all"] and len(args.logfiles) < 1:
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
    elif args.mode == "plot_best_to_date_avg_all":
        plot_best_to_date_avg_all(args)
    elif args.mode == "plot_variance_baseline":
        plot_variance_baseline(args)
    elif args.mode == "plot_error_ensemble":
        plot_error_ensemble(args)
    elif args.mode == "plot_mi_pid_mse":
        plot_mi_pid_mse(args)
    elif args.mode == "plot_mi_pid_mse_all":
        plot_mi_pid_mse_all(args)
    else:
        print("unknown mode '%s'" % (args.mode))
        
