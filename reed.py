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

from phidl import quickplot as qp
from phidl import set_quickplot_options
set_quickplot_options(blocking=True, show_ports=True)

import qnngds.utilities as qu
import qnngds.geometry as qg

def optimal_l(width = 1,
              side_length = 5,
              num_points = 50,
              layer = 1):
    """
    rewrite of pg.optimal_90deg to force the side length to specified value

    width       - width of wire in microns (optionally a tuple to make an asymmetric L)
    side_length - side length of L shape
    num_points  - number of points to use to create curve
    layer       - gds layer
    """

    D = Device("90deg")

    if isinstance(width, (int, float)):
        width = (width, width)

    # swap widths and ports to ensure width[1] >= width[0]
    p1, p2 = 1, 2
    if width[0] >= width[1]:
        # use >= so the default case uses the correct labeling scheme
        width = (width[1], width[0])
        p1, p2 = p2, p1

    a = 2 * width[0]
    log_vmax = np.log10(np.sinh((side_length/1.5-a/2)*np.pi/a))
    v = np.logspace(-log_vmax, log_vmax, num_points)
    xy = [(a/2*(1+2/np.pi*x),a/2*(1+2/np.pi*y)) for x,y in zip(np.arcsinh(1/v), np.arcsinh(v))]

    # add corners
    d = 1.5*xy[0][0]
    xy.append((width[0],d))
    xy.append((width[0]-width[1],d))
    xy.append((width[0]-width[1],0))
    xy.append((d,0))
    xy.append((d,width[0]))
    D.add_polygon([[x for x,y in xy], [y for x,y in xy]], layer=layer)
    D.add_port(name=p2, midpoint=[width[0]-width[1]/2, d], width=width[1], orientation=90)
    D.add_port(name=p1, midpoint=[d, width[0]/2], width=width[0], orientation=0)
    D.move((width[1]-width[0],0))
    return D

def optimal_tee(width = 1,
                side_length = 5,
                num_points = 50,
                layer = 1):
    """
    tee using optimal L's

    width       - width of wire in microns
    side_length - side length of L shape
    num_points  - number of points to use to create curve
    layer       - gds layer
    """
    D = Device("tee")

    if isinstance(width, (int, float)):
        width = (width, width)

    l_jct = optimal_l(width=width, side_length=side_length, layer=layer)
    jct_lower = D << l_jct
    jct_upper = D << l_jct
    jct_lower.mirror([0,0],[0,1])
    # connector to mirror jct
    conn = D << pg.connector(width=width[1])
    jct_lower.connect(jct_lower.ports[2], conn.ports[1])
    jct_upper.connect(jct_upper.ports[2], conn.ports[1])
    D = pg.union(D)
    D.flatten(single_layer=layer)
    D.add_port(name=1, port=jct_upper.ports[1])
    D.add_port(name=2, port=jct_lower.ports[1])
    D.add_port(name=3, port=conn.ports[2])
    return D

if __name__ == "__main__":
    # simple unit test
    D = Device("test")
    D << optimal_l(width=(1,1))
    D << optimal_l(width=(1,3))
    D << optimal_tee(width=(1,1))
    D << optimal_tee(width=(1,5))
    D.distribute(direction = 'y', spacing = 10)
    qp(D)
