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
    tee shape using optimal L's

    width       - width of wire in microns (optionally a tuple, second element sets the tee width)
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

def pad_array(num_pads = 8,
              workspace_size = 100,
              pad_size = (200, 250),
              pad_layers = (2,2,1,2,2,2,2,2),
              outline = 10,
              pos_tone = {1: False, 2: True}):
    """
    rewrite of qnngds pad_array
    
    renumbers ports so they follow standard ordering (CW increasing from top left corner)
    option to do negative tone for specific pads on specific layers
    num_pads        - total number of pads (will be evenly divided between the 4 sides
    workspace_size  - side length (in microns) of workspace area
    pad_size        - tuple, width and height (in microns) of each pad
    pad_layers      - tuple, gds layer for each pad (index 0 corresponds to the leftmost pad in the top row)
    outline         - outline width for positive tone
    pos_tone        - dictionary of whether or not to do positive tone for each layer
    """

    D = Device('pad_array')
    # check arguments
    fab_layers = set(pad_layers)
    if not fab_layers.issubset(set(pos_tone.keys())):
        ex_str = "pad_layers must only use values from pos_tone.keys()\nunique pad_layers = "
        ex_str += str(fab_layers) + "\n pos_tone keys = " + str(set(pos_tone.keys()))
        raise Exception(ex_str)

    min_pad_per_side, excess = divmod(num_pads, 4)
    pads_per_side = np.tile(min_pad_per_side, 4)
    for i in range(excess):
        pads_per_side[i] += 1
    pads_per_side.sort()
    conn_dict = {'W': pads_per_side[0],
                 'E': pads_per_side[1],
                 'S': pads_per_side[2],
                 'N': pads_per_side[3]}

    inner_compass = pg.compass_multi(size=(workspace_size, workspace_size), ports=conn_dict, layer=1)
    outer_compass_size = max(workspace_size, np.max(pads_per_side)*(min(pad_size) + outline*5))
    outer_compass = pg.compass_multi(size=(outer_compass_size, outer_compass_size), ports=conn_dict, layer=1)
   
    def port_idx(port_name, conn_dict, min_pads_per_side):
        # helper function to rename ports so they are ordered nicely
        side_keys = {'N': 0, 'E': 1, 'S': 2, 'W': 3}
        side_key = side_keys[port_name[0]]*(min_pads_per_side + 1)
        if port_name[0] in 'ES':
            side_key += conn_dict[port_name[0]] - int(port_name[1:])
        else:
            side_key += int(port_name[1:]) - 1
        return side_key

    inner_ports = sorted(inner_compass.get_ports(), key=lambda p: port_idx(p.name, conn_dict, min_pad_per_side))
    outer_ports = sorted(outer_compass.get_ports(), key=lambda p: port_idx(p.name, conn_dict, min_pad_per_side))
    final_ports = []
    for n, (p1, p2) in enumerate(zip(inner_ports, outer_ports)):
        T = Device('pad')
        workspace_connector = T << pg.straight(size=(p1.width/2-outline, p1.width/2), layer=pad_layers[n])
        workspace_connector.connect(workspace_connector.ports[1], p1)
        final_ports.append(workspace_connector.ports[1])
        pad = T << pg.straight(size=pad_size, layer=pad_layers[n])
        pad.connect(pad.ports[1], p2)
        T << pr.route_quad(workspace_connector.ports[2], pad.ports[1],
                           width1 = None, width2 = pad.ports[1].width/2, layer=pad_layers[n])
        T = pg.union(T, by_layer=True)
        T.add_port(name=1, port=workspace_connector.ports[1])
        if pos_tone[pad_layers[n]]:
            D << pg.outline(T, distance=outline, open_ports=True, layer=pad_layers[n])
        else:
            D << T
            for layer in fab_layers:
                if layer > pad_layers[n] and pos_tone[layer]:
                    # add etch to layers above pad_layer which are positive tone
                    etch = D << pg.straight(size=(pad_size[0]+2*outline, pad_size[1]+2*outline), layer=layer)
                    if np.dot(pad.ports[1].normal[1] - pad.ports[1].normal[0], np.array([1, 0])) != 0:
                        etch.rotate(90)
                    etch.move((pad.x - etch.x, pad.y - etch.y))
    D = pg.union(D, by_layer=True)
    for n, p in enumerate(final_ports):
        D.add_port(name=n, port=p)
    return D

if __name__ == "__main__":
    # simple unit test
    D = Device("test")
    #D << qg.pad_array(pad_iso=True, de_etch=True)
    #D << qg.pad_array(num=8, outline=10, layer=2)
    #D << pad_array(num_pads=22, workspace_size=1000, pad_layers=tuple(1 for i in range(22)), outline=10, pos_tone={1:True})
    #D << optimal_l(width=(1,1))
    #D << optimal_l(width=(1,3))
    #D << optimal_tee(width=(1,1))
    #D << optimal_tee(width=(1,5))
    D << pg.optimal_hairpin(width=1, pitch=1.2, length=5, turn_ratio=2, num_pts=100)
    D.distribute(direction = 'y', spacing = 10)
    qp(D)
