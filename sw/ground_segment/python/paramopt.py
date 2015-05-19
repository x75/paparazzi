#!/usr/bin/env python
"""Use hyperopt to optimize system parameters (e.g. PID params)"""

from __future__ import absolute_import, print_function

import argparse, sys, os, time

from ivy.std_api import *
sys.path.append(os.getenv("PAPARAZZI_HOME") + "/sw/lib/python")

from settings_tool import IvySettingsInterface

from ivy.ivy import IvyServer
    
class MyAgent(IvyServer):
    def __init__(self, name):
        IvyServer.__init__(self,'MyAgent')
        self.name = name
        self.start('127.255.255.255:2010')
        # self.bind_msg(self.handle_hello, 'hello .*')
        # self.bind_msg(self.handle_button, 'BTN ([a-fA-F0-9]+)')
        
    def handle_hello(self, agent):
        # print '[Agent %s] GOT hello from %r'%(self.name, agent)
        pass
      
    def handle_button(self, agent, btn_id):
        # print '[Agent %s] GOT BTN button_id=%s from %r'%(self.name, btn_id, agent)
        # let's answer!
        self.send_msg('BTN_ACK %s'%btn_id)
    
def main(args):
    print("main:", args)
    # ivyinterface = IvySettingsInterface(args.ac_id)
    
    index = 27
    # ivyinterface.lookup[index].value = 301
    # ivyinterface.SendSetting(index)
    # print("sent to ac %d idx: %d, value: %d" % (args.ac_id[0], index, ivyinterface.lookup[index].value))

    a = MyAgent("agent")

    time.sleep(3)
    print("sending ...")
    a.send_msg("DL_SETTING %d %d %d" % (args.ac_id[0], index, 317))
    print ("sent")
    a.send_msg("dl DL_SETTING %d %d %d" % (args.ac_id[0], index, 317))
    print ("sent")
    
    while True:
        time.sleep(0.1)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-a", "--ac_id", dest="ac_id", default=[167])
    
    args = parser.parse_args()
    main(args)
