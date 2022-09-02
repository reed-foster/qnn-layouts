# -*- coding: utf-8 -*-
"""
Created on Wed Mar 02 12:08:33 2022

@author: reedf

"""

import numpy as np
from phidl import Device
from phidl import Group
import phidl.geometry as pg
import phidl.routing as pr

import qnngds.utilities as qu
import qnngds.geometry as qg

def optimal_l(width = 1,
              side_length = 5,
              num_points = 50,
              layer = 1):
    """
    rewrite of pg.optimal_90deg to force the side length to specified value

    width       - width of wire in microns
    side_length - side length of L shape
    num_points  - number of points to use to create curve
    layer       - gds layer
    """

    D = Device("90deg")

    a = 2 * width
    log_vmax = np.log10(np.sinh((side_length/1.5-a/2)*np.pi/a))
    v = np.logspace(-log_vmax, log_vmax, num_points)
    xy = [(a/2*(1+2/np.pi*x),a/2*(1+2/np.pi*y)) for x,y in zip(np.arcsinh(1/v), np.arcsinh(v))]

    # add corners
    d = 1.5*xy[0][0]
    xy.append((width,d))
    xy.append((0,d))
    xy.append((0,0))
    xy.append((d,0))
    xy.append((d,width))
    D.add_polygon([[x for x,y in xy], [y for x,y in xy]], layer=layer)
    D.add_port(name=1, midpoint=[a/4, d], width=width, orientation=90)
    D.add_port(name=2, midpoint=[d, a/4], width=width, orientation=0)
    return D

def optimal_tee(width = 1,
                side_length = 5,
                num_points = 50,
                layer = 1):
    D = Device("tee")

    l_jct = optimal_l(width=width, side_length=side_length, layer=layer)
    jct_lower = D << l_jct
    jct_upper = D << l_jct
    # connector to mirror jct
    conn = D << pg.connector(width=width)
    jct_lower.connect(jct_lower.ports[1], conn.ports[1])
    jct_upper.connect(jct_upper.ports[2], conn.ports[1])
    D = pg.union(D)
    D.flatten(single_layer=layer)
    D.add_port(name=1, port=jct_upper.ports[1])
    D.add_port(name=2, port=jct_lower.ports[2])
    D.add_port(name=3, port=conn.ports[2])
    return D


