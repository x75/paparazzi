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
        # data collection etc
        self.cnt_eval = 0 # count number of evaluations of cost
        self.cnt_time = 0
        self.cnt_setting = 0
        self.maxsamp = 100
        self.numdata = 3 * 2 # sp, ref, measurement * 2 for attitude + batt, mode
        self.logdata = np.zeros((self.maxsamp, self.numdata))

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

        # data storage
        self.storage_version = "v2"
        self.tblfilename = "paramopt_ppz_v1.h5"
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
        print("done starting ivy (settings) for AC %s" % self.settings_if.GetACName())
        for group in self.settings_if.groups:
            print("Setting group %s" % group.name)
            for setting in group.member_list:
                print("%s (%d)" % (setting.shortname, setting.index))
        
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
        if not self.active:
            return

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
        elif msg.name == "ROTORCRAFT_STATUS":
            print("STATUS")
        elif msg.name == "RC":
            # print("RC")
            rc_vals = msg.fieldvalues[0].split(",")
            print("RC", rc_vals)
            if int(rc_vals[4]) > 1000:
                self.reset_params()
                self.active = False
                # self.cnt_time = 0

        # finish episode
        if self.cnt_time == self.maxsamp:
            # compute something
            self.active = False
            # self.cnt_time = 0
            # self.shutdown_handler(15, "")

    def reset_params(self):
        print("reset")
            
    def objective(self, params):
        # 29, 30, 31, 32, 33, 34, 35, 36
        # reset counter
        self.cnt_time = 0
        # reet logdata
        self.logdata[:,:] = 0.
        # play sound
        
        # suggest new setting
        self.settings_if.lookup[38].value = params[0] # 850 + np.random.randint(10)
        # send setting to autopilot
        self.settings_if.SendSetting(38)
        # tell agent about new setting
        self.cnt_setting += 1
        # evaluate
        self.active = True
        # print(objective)
        while self.active:
            time.sleep(1.)

        # compute cost
        # premature termination gives max cost
        if self.cnt_time != self.maxsamp:
            print("premature")
            cost = 1e4
        else:
            # compute cost: mean squared error of ref and est
            cost = np.mean(np.square(self.logdata[:,1] - self.logdata[:,2]))

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
        pl.cla()
        pl.plot(self.logdata)
            
        pl.show()
        pl.draw()
        return cost
    
def main(args):
    print("main:", args)
    # ivyinterface = IvySettingsInterface(args.ac_id)
    
    index = 27
    # ivyinterface.lookup[index].value = 301
    # ivyinterface.SendSetting(index)
    # print("sent to ac %d idx: %d, value: %d" % (args.ac_id[0], index, ivyinterface.lookup[index].value))

    # a = 

    pl.ion()
    
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
    a.settings_if.lookup[0].value = 14 # attitude_tune_x75
    # send setting to autopilot
    a.settings_if.SendSetting(0)

    
    
    while True:
        print("Starting episode")
        # print("Setting Att Loop pgain phi(%d) = %f" % (38, a.settings_if.lookup[38].value))
        # read current settings
        # for i in range(29, 38):
        #     print("settings.lookup[%d] (read) = %f" % (i, a.settings_if.lookup[i].value))
        params = (290 + np.random.randint(20), 0, 0, 0, 0, 0, 0, 0)
        cost = a.objective(params)
        print("cost %f" % cost)
        # a.interface.stop()
        time.sleep(5.)
        # a.interface.start()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-a", "--ac_id", dest="ac_id", default=167)
    
    args = parser.parse_args()
    main(args)
