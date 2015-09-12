#!/usr/bin/env python
"""Use hyperopt to optimize system parameters (e.g. PID params)"""

from __future__ import absolute_import, print_function

import argparse, sys, os, time, signal

from ivy.std_api import *
sys.path.append(os.getenv("PAPARAZZI_HOME") + "/sw/lib/python")

from settings_tool import IvySettingsInterface
from ivy_msg_interface import IvyMessagesInterface
from settings_tool import IvySettingsInterface
from pprz_msg.message import PprzMessage

from ivy.ivy import IvyServer

import numpy as np
import pylab as pl
    
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
            print("msg", msg.name)
            print("msg", msg.fieldnames)
            print("msg", msg.fieldvalues)
            print("msg", msg.fieldcoefs)
            # pass

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

        # finish episode
        if self.cnt_time == self.maxsamp:
            # compute something
            self.active = False
            self.cnt_time = 0
            pl.cla()
            pl.plot(self.logdata)
            
            pl.show()
            pl.draw()
            
            # self.shutdown_handler(15, "")
            
    def objective(self, params):
        # suggest new setting
        self.settings_if.lookup[38].value = 850 + np.random.randint(10)
        # send setting to autopilot
        self.settings_if.SendSetting(38)
        # tell agent about new setting
        self.cnt_setting += 1
        # evaluate
        self.active = True
        # print(objective)
        while self.active:
            time.sleep(1.)
        # print evaluation result
        print("cost %f" % np.mean(np.square(self.logdata[:,1] - self.logdata[:,2])))
    
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
    a.settings_if.lookup[0].value = 7
    # send setting to autopilot
    a.settings_if.SendSetting(0)
    
    while True:
        print("Starting episode")
        # print("Setting Att Loop pgain phi(%d) = %f" % (38, a.settings_if.lookup[38].value))
        # for i 
        print("settings (read):", a.settings_if.lookup[38].value)
        a.objective({})
        time.sleep(5.)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-a", "--ac_id", dest="ac_id", default=167)
    
    args = parser.parse_args()
    main(args)
