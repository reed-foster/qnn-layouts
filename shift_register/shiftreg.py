# -*- coding: utf-8 -*-
"""
Created on Wed Mar 02 14:11:16 2022

@author: reedf

"""

from __future__ import division, print_function, absolute_import
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

def shiftreg_halfstage(loop_sq = 2000,
                       loop_spacing = 5,
                       drain_sq = 200,
                       wire_w = 1.5,
                       constriction_w = 0.5,
                       nanowire_sq = 22,
                       switch_type = 'nw',
                       layer = 1):
    """
    single halfstage
    
    ntron or current-summing nanowire and series/loop inductor
    parameters:
    loop_sq         - number of loop inductor squares
    loop_spacing    - spacing between loop inductor and drain bias (in um)
    drain_sq        - number of drain inductor squares 
    wire_w          - width of gate, drain, source, and inductors (in um)
    constriction_w  - channel width (or width of nanowire constriction)
    nanowire_sq     - number of squares for nanowire
    switch_type     - 'ntron' or 'nw'
    layer           - layer
    """
    D = Device('halfstage')
    #####################################
    # create summing structure depending
    # on specified device type
    #####################################
    if switch_type == 'nw':
        # create constriction
        constriction_l = nanowire_sq*constriction_w
        nanowire = pg.straight(size=(constriction_w,constriction_l))
        nw = D << nanowire

        # taper above and below constriction
        taper_0 = pg.optimal_step(wire_w, constriction_w)
        taper_1 = pg.optimal_step(constriction_w, wire_w)
        t0 = D << taper_0
        t1 = D << taper_1
        t0.connect(taper_0.ports[2], nw.ports[1])
        t1.connect(taper_1.ports[1], nw.ports[2])

        # create junction for connecting gate and drain inductor to nanowire
        junction = pg.tee(size=(4*wire_w,wire_w), stub_size=(wire_w,wire_w), taper_type='fillet')
        jct = D << junction
        jct.connect(junction.ports[1], t0.ports[1])

        # create gate
        gate = pg.straight(size=(wire_w,1))
        g = D << gate
        g.connect(gate.ports[1], jct.ports[3])
    elif switch_type == 'ntron':
        ntron = qg.ntron_sharp(choke_w=constriction_w, choke_l=1, gate_w=wire_w,
                               channel_w=constriction_w, source_w=wire_w, drain_w=wire_w)
        nt = D << ntron
    else:
        raise ValueError(f'Invalid switch_type {switch_type} (choose: nw/ntron)')
    #####################################
    # create common structures
    #####################################
    # create drain inductor
    drain_ind = pg.snspd(wire_width = wire_w, wire_pitch = wire_w*2-0.002,
                         size = (None, wire_w*8), num_squares = drain_sq)
    ld = D << drain_ind
    if switch_type == 'nw':
        ld.connect(drain_ind.ports[1], jct.ports[2])
    else:
        ld.connect(drain_ind.ports[1], nt.ports['d'])

    # create junction for connecting bias and loop inductor to summing structure
    bias_junction = pg.tee(size=(4*wire_w,wire_w), stub_size=(wire_w,wire_w), taper_type='fillet')
    bjct = D << bias_junction
    bjct.connect(bias_junction.ports[2], ld.ports[2])

    # create drain bias
    drain = pg.straight(size=(wire_w,1))
    d = D << drain
    d.connect(drain.ports[1], bjct.ports[1])

    # create loop inductor
    ind_connector_0 = pg.straight(size=(wire_w, loop_spacing))
    ic0 = D << ind_connector_0
    ic0.connect(ind_connector_0.ports[1], bjct.ports[3])
    loop_ind = pg.snspd(wire_width = wire_w, wire_pitch = wire_w*2-0.002,
                        size = None, num_squares = loop_sq)
    ll = D << loop_ind
    ll.connect(loop_ind.ports[1], ic0.ports[2])
    ind_connector_1 = pg.straight(size=(wire_w, loop_spacing))
    ic1 = D << ind_connector_1
    ic1.connect(ind_connector_1.ports[1], ll.ports[2])
    
    # merge shapes and create new ports
    D = pg.union(D)
    D.flatten(single_layer=layer)
    if switch_type == 'nw':
        D.add_port(name='g', port=g.ports[2])
        D.add_port(name='s', port=t1.ports[2])
    else:
        D.add_port(name='g', port=nt.ports['g'])
        D.add_port(name='s', port=nt.ports['s'])
    D.add_port(name='d', port=d.ports[2])
    D.add_port(name='l', port=ic1.ports[2])
    D.name = 'nw_halfstage'
    D.info = locals()
    return D

def shiftreg_readout():
    pass

def shiftreg_input():
    pass

D = Device()

# D << qg.ntron(choke_w=0.15)
# D << qg.ntron_sharp(choke_w=0.5, choke_l=1, gate_w=1.5, channel_w=0.5, source_w=1.5, drain_w=1.5)
#D << shiftreg_halfstage(switch_type='ntron')
D << shiftreg_halfstage(switch_type='nw')
# D << pg.straight(size=(4,2))

qp(D)
