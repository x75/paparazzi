#!/usr/bin/env python
"""Use black box optimzer to tune system parameters (e.g. PID params)"""

from __future__ import absolute_import, print_function

import argparse, sys, os, time, signal

sys.path.append(os.getenv("PAPARAZZI_HOME") + "/sw/lib/python")

from ivy.std_api import *

from settings_tool import IvySettingsInterface
from ivy_msg_interface import IvyMessagesInterface
from settings_tool import IvySettingsInterface
from pprz_msg.message import PprzMessage

from ivy.ivy import IvyServer


def main(args):
    
    def shutdown_handler(signum, frame):
        print('Signal handler called with signal', signum)
        # s.isrunning = False
        # self.interface.shutdown()
        # del a
        sys.exit(0)

    # install interrupt handler
    signal.signal(signal.SIGINT, shutdown_handler)
    
    while True:
        print("blub")
        time.sleep(1.)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-a", "--ac_id", dest="ac_id", default=[167])
    
    args = parser.parse_args()
    main(args)
