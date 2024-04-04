# -*- coding: utf-8 -*-
"""
Created on Wed Mar 04 14:33:29 2024

@author: reedf

"""

import numpy as np
from phidl import Device
from phidl import Path
from phidl import Port
from phidl import CrossSection
import phidl.geometry as pg
import phidl.routing as pr
import phidl.path as pp
from phidl import quickplot as qp
from phidl import set_quickplot_options


def hTron_test_die(pad_size = 220,
                   fine_outline = 0.05,
                   coarse_outline = 25,
                   spacing = 200,
                   fine_layer = 1,
                   coarse_layer = 2):
    # sweep gate_width and channel_width from 15nm to 100nm
    gate_widths = np.array([0.1, 0.2, 0.05])
    channel_widths = np.array([0.4, 0.1, 0.2])
    gaps = np.array([0.01, 0.05, 0.025])
    #gate_widths = np.array([0.015, 0.025, 0.05])
    #channel_widths = np.array([0.015, 0.025, 0.05])
    #gaps = np.array([0.01, 0.05])
    # sweep gate length and channel length
    gate_squares = np.array([0])
    channel_extra_squares = np.array([0])
    devices = []
    for w_g in gate_widths:
        for w_c in channel_widths:
            if (w_g > w_c):
                # don't test devices where the gate is wider than the channel
                continue
            for g in gaps:
                for sq_g in gate_squares:
                    for sq_c in channel_extra_squares:
                        l_g = sq_g*w_g
                        l_c = l_g + sq_c*w_c
                        ht = hTron_pads(4*max(w_g,w_c), w_g, w_c, g, l_g, l_c, pad_size, fine_outline, coarse_outline, fine_layer, coarse_layer)
                        devices.append(ht)
    D = Device('hTron_test_die')
    n = np.ceil(len(devices)**0.5)
    digits = int(np.ceil(np.log10(n)))
    size = devices[0].size
    for i in range(4):
        P = Device('hTron_test_grid')
        nx = 0
        ny = 0
        x = 0
        y = 0
        for dev in devices:
            ht = P << dev
            dr = np.array([x, y])
            ht.move([-ht.xmin, -ht.ymin])
            ht.move(dr)
            t = P << pg.text(f'{chr(65+ny)}{nx:0{digits}d}', size=3*coarse_outline, layer=coarse_layer)
            t.move(-t.center)
            t.move(dr)
            t.move([size[0]/2, -2*coarse_outline])

            if nx == n - 1:
                nx = 0
                ny += 1
                x = 0
                y += size[1] + spacing
                #if (ny % 4) == 0:
                #    y += pad_size/2
            else:
                nx += 1
                x += size[0] + spacing
                #if (nx % 4) == 0:
                #    x += pad_size/2
        t = P << pg.text(f'{chr(65+i)}', size=1.5*pad_size, layer=coarse_layer)
        t.move(-t.center)
        t.move([x,y])
        t.move([pad_size,pad_size])
        P.move(-P.center)
        P.move((((i % 2) - 0.5)*(P.xsize + 2*spacing), ((i // 2) - 0.5)*(P.ysize + 2*spacing)))
        D << P
    D.move((5000,5000))
    return D

def hTron_pads(wire_width = 0.1,
               gate_width = 0.05,
               channel_width = 0.05,
               gap = 0.02,
               gate_length = 0.01,
               channel_length = 0.2,
               pad_size = 120,
               fine_outline = 0.05,
               coarse_outline = 25,
               fine_layer = 0,
               coarse_layer = 1):
    """Create a planar hTron with pads for wirebonding

    Parameters
    -----------------
    wire_width : int or float
        Width of routing wires in microns
    gate_width : int or float
        Width of superconducting gate in microns
    channel_width : int or float
        Width of superconducting channel in microns
    gap : int or float
        Spacing between gate and channel in microns
    gate_length : int or float
        Length of superconducting gate in microns
    channel_length : int or float
        Length of superconducting channel in microns
    pad_size : int or float
        Side length of pad in microns
    fine_outline : int or float
        Width of fine (low-current) positive tone outline in microns
    coarse_outline : int or float
        Width of coarse (high-current) positive tone outline in microns
    fine_layer : int
        Index of gds layer for fine outline
    coarse_layer : int
        Index of gds layer for coarse outline

    Returns
    -------------
    D : Device
        A Device containing a single hTron with pads

    """
    D = Device('hTron_pads')
    ht = Device('hTron')
    htron = ht << planar_hTron(wire_width=wire_width,
                               gate_width=gate_width,
                               channel_width=channel_width,
                               gap=gap,
                               gate_length=gate_length,
                               channel_length=channel_length)
    # taper htron wire_width up to the pad opening width
    X1 = CrossSection()
    length = coarse_outline
    start_width = wire_width
    end_width = 1.7*coarse_outline
    dwidth = end_width - start_width
    w = 0.5 + 0.5*(1 + 4/dwidth)**0.5
    widthfunc = lambda t: 1/(w - t) + start_width - 1/w
    X1.add(width=widthfunc, name='taper', ports=[1,2])
    P = pp.straight(length=length)
    taper = P.extrude(X1)
    for port in htron.ports:
        t = ht << taper
        t.connect(t.ports[1], htron.ports[port])
        ht.add_port(name=port, port=t.ports[2])
    # invert for positive tone resist
    htron_positive = D << pg.outline(ht, distance=fine_outline, open_ports=True, layer=fine_layer)
    # add pads
    P = Device('pads')
    center = P << pg.rectangle(size=(2*coarse_outline, 2*coarse_outline))
    center.move(-center.center)
    for p in range(4):
        pad = P << pg.rectangle(size=(pad_size,pad_size), layer=coarse_layer)
        position = (coarse_outline/2)*np.array([1 - 2*(p%2), 1 - 2*(p//2)])
        pad.move(-pad.bbox[0]+position)
        pad.rotate((p%2)*90*(1-2*(p//2))-90*(p//2), center=pad.bbox[0])
    D << pg.outline(P, distance=coarse_outline, layer=coarse_layer)
    label = f'w_g = {gate_width},\n w_c = {channel_width},\n g = {gap},\n l_g = {gate_length},\n l_c = {channel_length}'
    D.add_label(text=label, position=pad.center, layer=255)
    return D

def planar_hTron(wire_width = 0.2,
                 gate_width = 0.05,
                 channel_width = 0.05,
                 gap = 0.02,
                 gate_length = 0.01,
                 channel_length = 0.01,
                 outline = 0.05):
    """Create a planar hTron

    Parameters
    -----------------
    wire_width : int or float
        Width of routing wires in microns
    gate_width : int or float
        Width of superconducting gate in microns
    channel_width : int or float
        Width of superconducting channel in microns
    gap : int or float
        Spacing between gate and channel in microns
    gate_length : int or float
        Length of superconducting gate in microns
    channel_length : int or float
        Length of superconducting channel in microns
    outline : int or float
        Width of positive tone outline in microns

    Returns
    -------------
    D : Device
        A Device containing a single hTron

    """

    D = Device('hTron')

    ports = []
    outlines = []
    for direction,width,length in ((1,channel_width, channel_length), (-1,gate_width, gate_length)):
        W = Device('wire')
        constr = W << pg.straight(size=(width, np.max(length - 4*width,0)), layer=0)
        constr.move(-constr.center)
        constr.move([direction*(gap/2+width/2),0])
        taper = angled_taper(wire_width, width, 45)
        if direction < 0:
            taper.mirror()
        taper_lower = W << taper
        taper_lower.connect(taper_lower.ports[1], constr.ports[2])
        taper_upper = W << taper
        taper_upper.mirror()
        taper_upper.connect(taper_upper.ports[1], constr.ports[1])
        ports.append(taper_lower.ports[2])
        ports.append(taper_upper.ports[2])
        D << W

    D = pg.union(D)
    for p, port in enumerate(ports):
        D.add_port(name=p, port=port)
    return D

def angled_taper(wire_width = 0.2,
                 constr_width = 0.1,
                 angle = 60):
    """Create an angled taper with euler curves

    Parameters
    -----------------
    wire_width : int or float
        Width of wide end of taper
    constr_width: int or float
        Width of narrow end of taper
    angle: int or float
        Angle between taper ends in degrees

    Returns
    -------------
    D : Device
        A Device containing a single taper

    """

    D = Device('taper')

    # heuristic for length between narrow end and bend
    l_constr = constr_width*2 + wire_width*2
    # heuristic for length between wide end and bend
    l_wire = constr_width*2 + wire_width*2
    sin = np.sin(angle*np.pi/180)
    cos = np.cos(angle*np.pi/180)
    # path along the center of the taper
    p_center = np.array([[0,0], [l_constr, 0], [l_constr+l_wire*cos,l_wire*sin]])
    # upper (shorter) path along the inside edge of the taper
    p_upper = np.array([[0,constr_width/2], [0, constr_width/2], p_center[2] + [-wire_width/2*sin, wire_width/2*cos]])
    p_upper[1,0] = (constr_width/2-p_upper[2,1])*cos/sin + p_upper[2,0]
    # lower (longer) path along the outside edge of the taper
    p_lower = np.array([[0,-constr_width/2], [0, -constr_width/2], p_center[2] + [wire_width/2*sin, -wire_width/2*cos]])
    p_lower[1,0] = (-constr_width/2-p_lower[2,1])*cos/sin + p_lower[2,0]
    # interpolate euler curve between points
    P_upper = pp.smooth(points=p_upper, radius=wire_width, corner_fun=pp.euler, use_eff=False)
    P_lower = pp.smooth(points=p_lower, radius=wire_width, corner_fun=pp.euler, use_eff=False)
    
    # create a polygon
    points = np.concatenate((P_upper.points, P_lower.points[::-1]))
    D.add_polygon(points)

    # port 1: narrow/constr_width end, port 2: wide/wire_width end
    D.add_port(name=1, midpoint=(P_upper.points[0] + P_lower.points[0])/2, width=constr_width, orientation=180)
    D.add_port(name=2, midpoint=(P_upper.points[-1] + P_lower.points[-1])/2, width=wire_width, orientation=angle)

    return D

if __name__ == '__main__':
    set_quickplot_options(blocking=True, show_ports=True, new_window=True)
    D = hTron_test_die()
    D.flatten()
    D.write_gds('htron_tests.gds', unit=1e-6, precision=1e-9, auto_rename=True, max_cellname_length=1024, cellname='toplevel')
    qp(D)

