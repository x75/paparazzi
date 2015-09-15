from __future__ import print_function

import tables as tb

class PPRZ_Opt_Attitude(tb.IsDescription):
    # name = tb.StringCol(100)
    id = tb.Int64Col()
    pgain_phi   = tb.Int32Col()
    dgain_p     = tb.Int32Col()
    igain_phi   = tb.Int32Col()
    ddgain_p    = tb.Int32Col()
    pgain_theta = tb.Int32Col()
    dgain_q     = tb.Int32Col()
    igain_theta = tb.Int32Col()
    ddgain_q    = tb.Int32Col()
    mse        = tb.Float32Col()
    timeseries = tb.Float32Col(shape=(100, 12))

if __name__ == "__main__":
    print("class definition file only")
