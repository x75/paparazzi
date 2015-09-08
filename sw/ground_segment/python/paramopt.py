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
    
# class MyAgent(IvyServer):
class MyAgent(object):
    def __init__(self, name, msg_class = "telemetry"):
        # IvyServer.__init__(self,'MyAgent')
        # self.name = name
        # self.start('127.255.255.255:2010')
        # self.bind_msg(self.handle_hello, 'hello .*')
        # self.bind_msg(self.handle_button, 'BTN ([a-fA-F0-9]+)')

        # install interrupt handler
        signal.signal(signal.SIGINT, self.shutdown_handler)
        
        self.msg_class = msg_class
        self.interface = 0
        self.settings_if = 0
        # self.timer = threading.Timer(0.1, self.update_leds)
        # self.timer.start()

    def start(self):
        print("starting ivy ...")
        self.interface = IvyMessagesInterface(callback = self.message_recv,
                                              init = False)
        print("done starting ivy ...")
        
    def start_settings(self):
        print("starting ivy (settings) ...")
        self.settings_if = IvySettingsInterface([167])
        print("done starting ivy (settings) ...")
        
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
        if msg.msg_class == "telemetry":
            # print("msg", msg)
            pass
            
    
def main(args):
    print("main:", args)
    # ivyinterface = IvySettingsInterface(args.ac_id)
    
    index = 27
    # ivyinterface.lookup[index].value = 301
    # ivyinterface.SendSetting(index)
    # print("sent to ac %d idx: %d, value: %d" % (args.ac_id[0], index, ivyinterface.lookup[index].value))

    # a = 
    
    a = MyAgent("agent")

    print("agent created")
    
    # print("dir(a)", dir(a))
    a.start_settings()
    a.start()
        
    # sys.exit(0)
    # time.sleep(3)
    # print("sending ...")
    # a.send_msg("DL_SETTING %d %d %d" % (args.ac_id[0], index, 317))
    # print ("sent")
    # a.send_msg("dl DL_SETTING %d %d %d" % (args.ac_id[0], index, 317))
    # print ("sent")
    
    while True:
        print("blub")
        time.sleep(1.)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-a", "--ac_id", dest="ac_id", default=[167])
    
    args = parser.parse_args()
    main(args)
