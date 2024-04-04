# -*- coding: utf-8 -*-
"""
Created on Wed Mar 02 12:08:33 2022

@author: reedf

"""
import traceback

import numpy as np
from phidl import Device
from phidl import Path
from phidl import Port
import phidl.geometry as pg
import phidl.routing as pr
import phidl.path as pp

from phidl import quickplot as qp
from phidl import set_quickplot_options
#set_quickplot_options(blocking=True, show_ports=True)
set_quickplot_options(blocking=False, show_ports=True, new_window=True)

import qnngds.utilities as qu
import qnngds.geometry as qg


def shiftreg_stage(ind_sq = 500,
                   dev_outline = 0.2,
                   wire_w = 0.5,
                   ntron = None,
                   layer = 1):
    if ntron is None or not isinstance(ntron, Device):
        raise ValueError("Please pass a valid PHIDL object for the nTron geometry")
    if 'gate' in ntron.ports and 'drain' in ntron.ports and 'source' in ntron.ports:
        continue
    else:
        raise ValueError("nTron object does not have the expected ports")
