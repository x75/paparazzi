#!/usr/bin/env python
"""Use hyperopt to optimize system parameters (e.g. PID params)"""

from __future__ import absolute_import, print_function

import argparse, sys, os, time, signal

from ivy.std_api import *
sys.path.append(os.getenv("PAPARAZZI_HOME") + "/sw/lib/python")

from settings_tool import IvySettingsInterface
from ivy_msg_interface import IvyMessagesInterface
# from settings_tool import IvySettingsInterface
from pprz_msg.message import PprzMessage

from ivy.ivy import IvyServer

import numpy as np
import pylab as pl
import tables as tb

from paramopt_data import PPRZ_Opt_Attitude

import pygame

# hyperopt stuff
from hyperopt import hp, fmin, tpe, Trials, rand, STATUS_OK, STATUS_FAIL, anneal
import hp_gpsmbo.hpsuggest
from hp_gpsmbo import suggest_algos
from functools import partial
    
# class MyAgent(IvyServer):
class MyAgent(object):
    """Parameter optimizing agent for Paparazzi system"""
    def __init__(self, name, ac_id, msg_class = "telemetry"):
        # IvyServer.__init__(self,'MyAgent')
        # self.name = name
        # self.start('127.255.255.255:2010')
        # self.bind_msg(self.handle_hello, 'hello .*')
        # self.bind_msg(self.handle_button, 'BTN ([a-fA-F0-9]+)')

        # install interrupt handler
        signal.signal(signal.SIGINT, self.shutdown_handler)

        # aircraft ID
        self.ac_id = [ac_id]

        # message class for filtering
        self.msg_class = msg_class
        # settings interface, this one does the  Ivy init
        self.settings_if = 0
        # message interface, so this one doesn't need to do it, see below
        self.interface = 0
        
        # self.timer = threading.Timer(0.1, self.update_leds)
        # self.timer.start()

        # activity control
        self.active = False
        self.safety = False
        # data collection etc
        self.cnt_eval = 0 # count number of evaluations of cost
        self.cnt_time = 0
        self.cnt_setting = 0
        self.maxsamp = 200
        self.numdata = 3 * 2 + 6 # sp, ref, measurement * 2 for attitude + 6 RC channels
        self.logdata = np.zeros((self.maxsamp, self.numdata))

        # parameters optimized
        if self.ac_id[0] == 167:
            # self.param_idx = [29, 30, 31, 32, 33, 34, 35, 36]
            self.param_idx = range(29, 37)
        elif self.ac_id[0] == 201:
            self.param_idx = range(38, 46)
        print("param indices", self.param_idx)
        # backup parameters
        self.default_params = (
            300, # pgain_phi
            350, # dgain p
            100, # igain_phi
            150, # ddgain_p
            300, # pgain_theta
            350, # dgain q
            100, # igain_theta
            150 # ddgain_q
            )
        print("default_params:", self.default_params)
        # self.onboard_params = (-1, -1, -1, -1, -1, -1, -1, -1)
        self.onboard_params = [-1, -1, -1, -1, -1, -1, -1, -1]
        self.desired_params = self.default_params
        self.params_eq = False

        # data storage
        self.storage_version = "v2"
        self.tblfilename = "paramopt_ppz_v2.h5"
        self.h5file = tb.open_file(self.tblfilename, mode = "a")
        # check if top group exists
        try:
            self.g1    = self.h5file.get_node("/%s" % self.storage_version)
            # self.table = self.h5file.list_nodes("/v1")[0]
            self.table = self.h5file.get_node("/%s/evaluations" % self.storage_version)
        except:
            self.g1     = self.h5file.create_group(self.h5file.root, self.storage_version,
                                               "Optimization run params, perf and logdata")
            if self.storage_version == "v2":
                self.table = self.h5file.create_table(self.g1, 'evaluations', PPRZ_Opt_Attitude, "Single optimizer evaluations")
        print(self.g1, self.table)
        self.pprz_opt_attitude = self.table.row

    def start(self):
        print("starting ivy ...")
        self.interface = IvyMessagesInterface(callback = self.message_recv,
                                              init = False)
        print("done starting ivy ...")
        
    def start_settings(self):
        print("starting ivy (settings) ...")
        self.settings_if = IvySettingsInterface(self.ac_id)
        self.settings_if.RegisterCallback(self.settings_update_cb)
        print("done starting ivy (settings) for AC %s" % self.settings_if.GetACName())
        for group in self.settings_if.groups:
            print("Setting group %s" % group.name)
            for setting in group.member_list:
                print("%s (%d)" % (setting.shortname, setting.index))

    def settings_update_cb(self, sidx, sval, frem):
        # print("settings_update_cb", sidx, sval, frem)
        pass
        # if self.safety:
        #     fsval = float(sval)
        #     isval = int(fsval)
        #     if sidx in self.param_idx:
        #         if isval != self.default_params[sidx-29]:
        #             print("FAIL", sidx, isval, self.default_params[sidx-29])
        #             self.reset_params()
        
    def shutdown_handler(self, signum, frame):
        print('Signal handler called with signal', signum)
        # s.isrunning = False
        self.interface.shutdown()
        # del a
        sys.exit(0)
        # raise IOError("Couldn't open device!")
    
    # def handle_hello(self, agent):
    #     # print '[Agent %s] GOT hello from %r'%(self.name, agent)
    #     pass
      
    # def handle_button(self, agent, btn_id):
    #     # print '[Agent %s] GOT BTN button_id=%s from %r'%(self.name, btn_id, agent)
    #     # let's answer!
    #     self.send_msg('BTN_ACK %s'%btn_id)

    def message_recv(self, ac_id, msg):
        # only show messages of the requested class
        # if msg.msg_class != self.msg_class:
        #     return
        # if ac_id in self.aircrafts and msg.name in self.aircrafts[ac_id].messages:
        # if time.time() - self.aircrafts[ac_id].messages[msg.name].last_seen < 0.2:
        # return
        # print("ac_id", ac_id)
        # if msg.name.startswith("DL_VALUE"):
        if msg.name ==  "DL_VALUES":
            # print(msg.name, msg.fieldvalues[1])
            # sys.stdout.write("DL_VALUES ")
            for idx in self.param_idx:
                r_idx = idx - self.param_idx[0]
                # print("msg", msg.fieldvalues[1])
                dl_values = msg.fieldvalues[1].split(",")
                # sys.stdout.write("'%s', " % (dl_values[idx]))
                if dl_values[idx] != '?':
                    self.onboard_params[r_idx] = float(dl_values[idx])
                else:
                    self.onboard_params[r_idx] = -1.
            # sys.stdout.write("\n")
                    
        # return
        
        if msg.msg_class == "telemetry":
            # print("msg", dir(msg))
            # print("msg", msg.name)
            # print("msg", zip(msg.fieldnames, msg.fieldvalues))
            # print("msg", msg.fieldcoefs)
            pass
        else:
            return

        # print(type(msg.fieldvalues[0]))

        # stash and save logdata
        if msg.name == "RC":
            # print("RC")
            rc_vals = msg.fieldvalues[0].split(",")
            rc_vals_len = min(len(rc_vals), 6)
            # print("RC", len(rc_vals), rc_vals)
            if self.cnt_time < self.maxsamp:
                self.logdata[self.cnt_time,6:6+rc_vals_len] = rc_vals[0:rc_vals_len]
            if int(rc_vals[4]) > 1000:
                self.active = False
                if not self.safety:
                    self.safety = True
                    self.reset_params()
                    # time.sleep(0.5)
                    # self.reset_params()
                # self.cnt_time = 0
            elif int(rc_vals[4]) < -1000:
                self.safety = False
            return # can only be one message

        if not self.active:
            return
                
        if msg.name == "STAB_ATTITUDE_INT":
            self.logdata[self.cnt_time,2] = float(msg.fieldvalues[3]) * msg.fieldcoefs[3] # est_phi
            self.logdata[self.cnt_time,5] = float(msg.fieldvalues[4]) * msg.fieldcoefs[4] # est_theta
            # increment counter
            self.cnt_time += 1
            
        elif msg.name == "STAB_ATTITUDE_REF_INT":
            self.logdata[self.cnt_time,0] = float(msg.fieldvalues[0]) * msg.fieldcoefs[0] # sp_phi
            self.logdata[self.cnt_time,1] = float(msg.fieldvalues[3]) * msg.fieldcoefs[3] # ref_phi
            self.logdata[self.cnt_time,3] = float(msg.fieldvalues[1]) * msg.fieldcoefs[1] # sp_theta
            self.logdata[self.cnt_time,4] = float(msg.fieldvalues[4]) * msg.fieldcoefs[4] # ref_theta
        # elif msg.name == "ROTORCRAFT_STATUS":
        #     print("STATUS")

        # finish episode
        if self.cnt_time == self.maxsamp:
            # compute something
            self.active = False
            # self.cnt_time = 0
            # self.shutdown_handler(15, "")

    def set_param(self, idx, val):
        self.settings_if.lookup[idx].value = str(val)
        self.settings_if.SendSetting(idx)
        # for i in range(10):
        #     print("val check", val, self.settings_if.lookup[idx].value)
        #     time.sleep(0.2)

    def set_params(self, params):
        self.params_eq = False
        # for idx in self.param_idx:
        #     r_idx = idx - self.param_idx[0]
        #     self.set_param(idx, params[r_idx])

        while not self.params_eq:
            print("set_params", self.params_eq)
            for idx in self.param_idx:
                r_idx = idx - self.param_idx[0]
                self.set_param(idx, params[r_idx])
                print(idx, params)
                print("params[%d]: %f == %f?" % (idx, self.settings_if.lookup[idx].value, self.onboard_params[r_idx]))
                print()
                # if self.settings_if.lookup[idx].value == params[r_idx]:
                if self.settings_if.lookup[idx].value == self.onboard_params[r_idx]:
                    print("################################################################################")
                    self.params_eq = True
                else:
                    self.params_eq = False
            time.sleep(0.1)

    def set_params_cont(self):
        for idx in self.param_idx:
            r_idx = idx - self.param_idx[0]
            self.set_param(idx, self.desired_params[r_idx])
            # time.sleep(0.05)
                        
    def reset_params(self):
        print("reset")
        # self.set_params(self.default_params)
        self.desired_params = self.default_params
        
        # for idx in self.param_idx:
        #     r_idx = idx - self.param_idx[0]
        #     while self.settings_if.lookup[idx].value != self.default_params[r_idx]:
        #         self.set_param(idx, int(self.default_params[r_idx]))
        #         # print("UNSAME")
        #         # self.settings_if.lookup[idx].value = int(self.default_params[r_idx]) # 850 + np.random.randint(10)
        #         # self.settings_if.SendSetting(idx)


        # time.sleep(0.1)
        # # check
        # for idx in self.param_idx:
        #     r_idx = idx - self.param_idx[0]
        #     if self.settings_if.lookup[idx].value != self.default_params[r_idx]:
        #         self.settings_if.lookup[idx].value = self.default_params[r_idx]
        #         self.settings_if.SendSetting(idx)
        #         print("UNSAME: params[%d] = %d" % (idx, self.settings_if.lookup[idx].value))        

    def print_params(self):
        for idx in self.param_idx:
            r_idx = idx - self.param_idx[0]
            # print("r_idx = %d" % r_idx)
            # print(self.settings_if.lookup[idx].value, )
            sys.stdout.write("%s, " % self.settings_if.lookup[idx].value)
            sys.stdout.write("%s, " % self.onboard_params[r_idx])
        sys.stdout.write("\n")
        
    def clean_logdata(self, logdata):
        rows, cols = logdata.shape
        for j in range(cols):
            for i in range(1, rows):
                if logdata[i,j] == 0.:
                    logdata[i,j] = logdata[i-1,j]
        return logdata
                    
    def objective(self, params):
        print("############################################################")
        print("cnt %d" % (self.cnt_eval))
        self.cnt_eval += 1
        print("params:", params)
        
        # # FIXME: if symmetry hack
        # params_new = (params[0], params[1], params[2], params[3],
        #               params[0], params[1], params[2], params[3])
        # params = params_new
        
        # 29, 30, 31, 32, 33, 34, 35, 36
        assert(len(params) == len(self.default_params))
        # status
        status = STATUS_FAIL
        # reset counter
        self.cnt_time = 0
        # reet logdata
        self.logdata[:,:] = 0.
        # play sound

        # check for existing data
        # print("query tuple", tuple([params[i] for i in range(8)]))
        query = """(pgain_phi == %d) & (dgain_p == %d) & \
        (igain_phi == %d) & (ddgain_p == %d) & \
        (pgain_theta == %d) & (dgain_q == %d) & \
        (igain_theta == %d) & (ddgain_q == %d)""" % tuple([params[i] for i in range(8)])
        print("query = %s" % (query))
        existing_run_data = \
        [ (x["pgain_phi"],   x["igain_phi"],   x["dgain_p"], x["ddgain_p"],
           x["pgain_theta"], x["igain_theta"], x["dgain_q"], x["ddgain_q"],
           x["mse"]) for x in self.table.where(query)]

        print("existing_run_data", existing_run_data)

        # FIXME: if mode manual        
        if len(existing_run_data) > 0:
            cost = existing_run_data[-1][-1]
            print("reusing existing run data: mse = %f" % (cost))
            status = STATUS_OK
            time.sleep(0.1)
            # return cost
            return {"loss": cost, "status": status, "loss_variance": 6.}
        

        # play sound when starting new eval
        pygame.mixer.music.play()
        # suggest new setting
        self.desired_params = params
        for i in range(3):
            self.set_params_cont()
            # time.sleep(0.1)
        
        # for idx in self.param_idx:
        #     r_idx = idx - self.param_idx[0]
        #     # print("r_idx = %d" % r_idx)
        #     # self.settings_if.lookup[idx].value = params[r_idx]
        #     # print("params[%d] = %d" % (idx, self.settings_if.lookup[idx].value))
        #     # 850 + np.random.randint(10)
        #     # send setting to autopilot
        #     # self.settings_if.SendSetting(idx)
        #     self.set_param(idx, int(params[r_idx]))
            
        # tell agent about new setting
        self.cnt_setting += 1
        # evaluate
        self.active = True
        # print(objective)
        while self.active:
            # continuously set parameters
            self.set_params_cont()
            self.print_params()
            time.sleep(1.)

        while self.safety:
            print("Waiting for safety")
            self.set_params_cont()
            self.print_params()
            time.sleep(1)

        # clean logdata
        self.clean_logdata(self.logdata)


        # reset params to default while waiting for next run
        self.desired_params = self.default_params
        for i in range(3):
            self.set_params_cont()
                
        # compute cost
        # premature termination gives max cost
        if self.cnt_time != self.maxsamp:
            print("premature")
            cost = 1e4
        else:
            # compute cost: mean squared error of ref and est
            c_phi   = np.mean(np.square(self.logdata[:,1] - self.logdata[:,2]))
            c_theta = np.mean(np.square(self.logdata[:,4] - self.logdata[:,5]))
            # FIXME: include p,q cost
            cost = c_phi + c_theta
            status = STATUS_OK

        # save data to pytable
        ts = time.strftime("%Y%m%d%H%M%S")
        self.pprz_opt_attitude["id"] = int(ts)
        self.pprz_opt_attitude["pgain_phi"]   = params[0]
        self.pprz_opt_attitude["dgain_p"]     = params[1]
        self.pprz_opt_attitude["igain_phi"]   = params[2]
        self.pprz_opt_attitude["ddgain_p"]    = params[3]
        self.pprz_opt_attitude["pgain_theta"] = params[4]
        self.pprz_opt_attitude["dgain_q"]     = params[5]
        self.pprz_opt_attitude["igain_theta"] = params[6]
        self.pprz_opt_attitude["ddgain_q"]    = params[7]
        # set run performance measure
        self.pprz_opt_attitude["mse"]         = cost
        # set run logdata
        # self.pprz_opt_attitude["timeseries"][0:self.maxsamp]  = self.logdata
        self.pprz_opt_attitude["timeseries"]  = self.logdata
        # append new row
        self.pprz_opt_attitude.append()
        self.table.flush()

        # print evaluation result
        pl.subplot(211)
        pl.cla()
        pl.plot(self.logdata[:,0:6])
            
        pl.subplot(212)
        pl.cla()
        pl.plot(self.logdata[:,6:12])
        
        pl.show()
        pl.draw()
        print("cost = %f" % cost)
        # time.sleep(1.)
        # return cost
        return {"loss": cost, "status": status, "loss_variance": 6.}
    
def main_loop_manual(args, a):
    # manual testing main loop
    # while True:
    for i in range(int(args.maxeval)):
        print("Starting episode")
        # print("Setting Att Loop pgain phi(%d) = %f" % (38, a.settings_if.lookup[38].value))
        # read current settings
        # for i in range(29, 38):
        #     print("settings.lookup[%d] (read) = %f" % (i, a.settings_if.lookup[i].value))
        params = (
            290 + np.random.randint(20),
            340 + np.random.randint(20),
            90 + np.random.randint(20),
            140 + np.random.randint(20),
            290 + np.random.randint(20),
            340 + np.random.randint(20),
            90 + np.random.randint(20),
            140 + np.random.randint(20),
        )
        params = a.default_params
        cost = a.objective(params)
        print("cost %f" % cost["loss"])
        # a.interface.stop()
        time.sleep(3.)
        # a.interface.start()
    a.shutdown_handler(15, 0)

def main_loop_hyperopt(args, a):
    # hyperopt main loop
    # full space
    # space = [
    #     hp.quniform("pgain_phi", 10, 1000, 1),
    #     hp.quniform("dgain_p", 0, 500, 1),
    #     hp.quniform("igain_phi", 0, 400, 1),
    #     hp.quniform("ddgain_p", 10, 500, 1),
    #     hp.quniform("pgain_theta", 10, 1000, 1),
    #     hp.quniform("dgain_q", 0, 500, 1),
    #     hp.quniform("igain_theta", 0, 400, 1),
    #     hp.quniform("ddgain_q", 10, 500, 1),
    #     ]
    # restricted space
    space = [
        hp.quniform("pgain_phi", 100, 500, 1),
        hp.quniform("dgain_p", 150, 500, 1),
        hp.quniform("igain_phi", 10, 200, 1),
        hp.quniform("ddgain_p", 10, 300, 1),
        hp.quniform("pgain_theta", 100, 500, 1),
        hp.quniform("dgain_q", 150, 500, 1),
        hp.quniform("igain_theta", 10, 200, 1),
        hp.quniform("ddgain_q", 10, 300, 1)
        ]
    
    # # more retricted space / symmetry hack
    # space = [
    #     hp.quniform("pgain", 100, 600, 1),
    #     hp.quniform("dgain", 150, 600, 1),
    #     hp.quniform("igain", 10, 250, 1),
    #     hp.quniform("ddgain", 10, 300, 1)
    #     ]
        
    trials = Trials()
    if args.mode == "hpo_tpe":
        suggest = tpe.suggest
    elif args.mode == "hpo_anneal":
        suggest = anneal.suggest
    elif args.mode == "hpo_gp_ucb":
        suggest = partial(suggest_algos.ucb, stop_at=1.)
        suggest = suggest
    elif args.mode == "hpo_gp_ei":
        suggest = partial(suggest_algos.ei, stop_at=1.)
        suggest = suggest
    else: # hpo_rand
        suggest = rand.suggest

    best = fmin(a.objective,
                space,
                algo=suggest,
                max_evals=int(args.maxeval),
                rstate=np.random.RandomState(123), # 1, 10, 123, 
                trials=trials)
    print("best", best)
    a.shutdown_handler(15, 0)
    
def main(args):
    print("main:", args)
    # ivyinterface = IvySettingsInterface(args.ac_id)
    
    index = 27
    # ivyinterface.lookup[index].value = 301
    # ivyinterface.SendSetting(index)
    # print("sent to ac %d idx: %d, value: %d" % (args.ac_id[0], index, ivyinterface.lookup[index].value))

    # a = 

    pl.ion()


    pygame.init()

    pygame.mixer.music.load("paramopt_1.wav")
    
    a = MyAgent("agent", int(args.ac_id))

    print("agent created")
    
    # print("dir(a)", dir(a))
    a.start_settings()
    a.start()

    # wait for ivy to come up
    time.sleep(3.)
            
    # sys.exit(0)
    # time.sleep(3)
    # print("sending ...")
    # a.send_msg("DL_SETTING %d %d %d" % (args.ac_id[0], index, 317))
    # print ("sent")
    # a.send_msg("dl DL_SETTING %d %d %d" % (args.ac_id[0], index, 317))
    # print ("sent")

    # set telemetry mode to attitude loop
    # a.settings_if.lookup[0].value = 7  # attitude_loop
    a.settings_if.lookup[0].value = 12 # attitude_tune_x75
    # send setting to autopilot
    a.settings_if.SendSetting(0)

    if args.mode == "manual":
        main_loop_manual(args, a)
    elif args.mode.startswith("hpo"):
        main_loop_hyperopt(args, a)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-a", "--ac_id", dest="ac_id", default=167)
    parser.add_argument("-m", "--mode", dest="mode", default="manual")
    parser.add_argument("-me", "--maxeval", default=60)
    
    args = parser.parse_args()
    main(args)
