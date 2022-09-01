# -*- coding: utf-8 -*-
"""
Created on Wed Aug 31 14:00:00 2022

@author: reedf

"""

import numpy as np
from phidl import Device
from phidl import Group
import phidl.geometry as pg
import phidl.routing as pr
import phidl.path as pp
from phidl import quickplot as qp
from phidl import set_quickplot_options
set_quickplot_options(blocking=True, show_ports=True)
#set_quickplot_options(blocking=True, show_ports=False)

import qnngds.utilities as qu
import qnngds.geometry as qg

def optimal_l(width = 1,
              side_length = 5,
              num_points = 20,
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
                num_points = 20,
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

def shiftreg_halfstage(ind_sq = 500,
                       ind_spacing = 3,
                       dev_outline = 0.2,
                       wire_w = 1,
                       constriction_w = 0.280,
                       routing_w = 5,
                       drain_sq = 5,
                       source_sq = 5,
                       gate = True,
                       layer = 1):
    """
    shift register halfstage

    vertical column with ntron, optional input, shift_clk, and shunt followed by loop inductor
    parameters:
    ind_sq          - number of squares in loop inductor
    ind_spacing     - spacing between ntron column and loop inductor in microns (prevent heating of loop)
    dev_outline     - outline width in microns (all devices are outlined for positive tone)
    wire_w          - width of wire in microns
    constriction_w  - width of channel/gate constriction in microns
    routing_w       - width of routing wires in microns
    drain_sq        - number of squares on drain side
    source_sq       - number of squares on source side
    gate            - True if gate should be included, False otherwise
    layer           - gds layer
    """

    D = Device("sr_halfstage")

    #######################################################
    # create ntron
    #######################################################
    source_taper = pg.taper(source_sq*wire_w - constriction_w/2, wire_w, constriction_w, layer=layer)
    drain_taper = pg.taper(drain_sq*wire_w - constriction_w/2, wire_w, constriction_w, layer=layer)
    # channel is 1 square
    channel = pg.compass(size=(constriction_w, constriction_w), layer=layer)
    gnd_taper = qg.hyper_taper(wire_w, routing_w, wire_w, layer=layer)
    ch = D << channel
    st = D << source_taper
    dt = D << drain_taper
    gnd = D << gnd_taper
    st.connect(st.ports[2], ch.ports['S'])
    dt.connect(dt.ports[2], ch.ports['N'])
    gnd.connect(gnd.ports[1], st.ports[1])
    if gate:
        gate_taper = pg.taper(wire_w - constriction_w/2, wire_w, constriction_w, layer=layer)
        gt = D << gate_taper
        gt.connect(gt.ports[2], ch.ports['W'])

    #######################################################
    # create tees for shift_clk and shunt
    #######################################################
    tee_jct = optimal_tee(width=wire_w, side_length=5*wire_w, layer=layer)
    bias_jct = D << tee_jct
    shunt_jct = D << tee_jct
    bias_jct.connect(bias_jct.ports[2], dt.ports[1])
    shunt_jct.connect(shunt_jct.ports[1], bias_jct.ports[3])

    #######################################################
    # make loop inductor
    #######################################################
    ind_pitch = wire_w*2-(wire_w-2*dev_outline)
    #ind_height = (3+np.floor((source_sq+drain_sq+1+5)*wire_w/ind_pitch))*ind_pitch
    if ind_sq > 500:
        ind_width = 50*wire_w
    elif ind_sq > 200:
        ind_width = 30*wire_w
    else:
        ind_width = 30*wire_w
    loop_ind = pg.snspd(wire_width=wire_w, wire_pitch=ind_pitch, size=(ind_width, None),#size=(None, ind_height),
                        num_squares=ind_sq, layer=layer)
    ind_spacer = pg.straight(size=(wire_w, (ind_spacing-4*wire_w)), layer=layer)
    ind_sp = D << ind_spacer
    ind = D << loop_ind
    ind_sp.connect(ind_sp.ports[1], bias_jct.ports[3])
    ind.connect(ind.ports[1], ind_sp.ports[2])

    #######################################################
    # add snake to connect previous stage to gate input
    #######################################################
    if gate:
        leftover_height = ind.ysize - (drain_sq+5)*wire_w + 0.5*wire_w
        snake = False
        min_side_length = 8*wire_w
        if leftover_height < -min_side_length:
            up = True
            snake = True
        elif leftover_height > min_side_length:
            up = False
            snake = True
        else:
            lower = D << pg.connector(midpoint=(gt.ports[1].x-10*wire_w,ind.ports[2].y),width=wire_w)
            r1 = pr.route_smooth(lower.ports[1], gt.ports[1], length1=3*wire_w, length2=3*wire_w, path_type='Z')
            r2 = pg.copy(r1).move([-wire_w,0])
            rcomb = pg.boolean(A=r1, B=r2, operation='or', precision=1e-6, layer=layer)
            D << rcomb
        if snake:
            spacer_height = abs(leftover_height) - min_side_length
            half_snake = optimal_l(width=wire_w, side_length=min_side_length/2, layer=layer)
            upper = D << half_snake
            lower = D << half_snake
            upper.connect(upper.ports[2 if up else 1], gt.ports[1])
            snake_sp = D << pg.straight(size=(wire_w, spacer_height), layer=layer)
            snake_sp.connect(snake_sp.ports[1], upper.ports[1 if up else 2])
            lower.connect(lower.ports[1 if up else 2], snake_sp.ports[2])

    #######################################################
    # finalize device
    #######################################################
    D = pg.union(D)
    D.flatten(single_layer=layer)
    if gate:
        if snake:
            D.add_port(name='input', port=lower.ports[2 if up else 1])
        else:
            D.add_port(name='input', port=lower.ports[2])
    D.add_port(name='gnd', port=gnd.ports[2])
    D.add_port(name='shift_clk', port=bias_jct.ports[1])
    D.add_port(name='shunt', port=shunt_jct.ports[3])
    D.add_port(name='output', port=ind.ports[2])
    return D

def multistage_shiftreg(ind_spacings = (6, 3, 6),
                        readout_channel_w = 0.24,
                        readout_gate_w = 0.035,
                        readout_source_sq = 5,
                        readout_drain_sq = 5,
                        readout_gate_sq = 10,
                        ind_sq = 500,
                        dev_outline = 0.2,
                        wire_w = 1,
                        constriction_w = 0.280,
                        routing_w = 5,
                        drain_sq = 5,
                        source_sq = 5,
                        layer = 1):
    """
    shift register
    
    cascaded half stages, each with individually settable inductor spacing
    parameters:
    ind_spacings        - spacing between ntron column and loop inductor in microns for each stage
    readout_channel_w   - width of readout ntron channel in microns
    readout_gate_w      - width of readout ntron gate in microns
    readout_source_sq   - number of squares for source of readout ntron 
    readout_drain_sq    - number of squares for drain of readout ntron 
    readout_gate_sq     - number of squares for gate of readout ntron 
    ind_sq              - number of squares in loop inductor
    dev_outline         - outline width in microns (all devices are outlined for positive tone)
    wire_w              - width of wire in microns
    constriction_w      - width of channel/gate constriction in microns
    routing_w           - width of routing wires in microns
    drain_sq            - number of squares on drain side
    source_sq           - number of squares on source side
    layer               - gds layer
    """

    D = Device('multistage_shiftreg')

    #######################################################
    # make halfstages
    #######################################################
    halfstages = []
    for i, ind_spacing in enumerate(ind_spacings):
        halfstages.append(D << shiftreg_halfstage(ind_sq=ind_sq, ind_spacing=ind_spacing,
                                                  dev_outline=dev_outline, wire_w=wire_w,
                                                  constriction_w=constriction_w,
                                                  routing_w=routing_w, drain_sq=drain_sq,
                                                  source_sq=source_sq, gate=(i>0), layer=layer))
        if i > 0:
            halfstages[-1].connect(halfstages[-1].ports['input'], halfstages[-2].ports['output'])
    
    #######################################################
    # make readout ntron
    #######################################################
    tee_jct = optimal_tee(width=wire_w, side_length=5*wire_w, layer=layer)
    output_jct = D << tee_jct
    output_jct.connect(output_jct.ports[1], halfstages[-1].ports['output'])

    source_taper = pg.taper(readout_source_sq*wire_w - readout_channel_w/2, wire_w, readout_channel_w, layer=layer)
    drain_taper = pg.taper(readout_drain_sq*wire_w - readout_channel_w/2, wire_w, readout_channel_w, layer=layer)
    # channel is 1 square
    channel = pg.compass(size=(readout_channel_w, readout_channel_w), layer=layer)
    gate_taper = pg.taper(readout_gate_sq*wire_w - readout_channel_w/2 - readout_gate_w/2, wire_w, readout_gate_w, layer=layer)
    gate_choke = pg.straight(size=(readout_gate_w,readout_gate_w), layer=layer)
    gnd_taper = qg.hyper_taper(wire_w, routing_w, wire_w, layer=layer)
    st = D << source_taper
    dt = D << drain_taper
    ch = D << channel
    gt = D << gate_taper
    gc = D << gate_choke
    gnd = D << gnd_taper
    dt.connect(dt.ports[1], output_jct.ports[2])
    ch.connect(ch.ports['W'], dt.ports[2])
    st.connect(st.ports[2], ch.ports['E'])
    gc.connect(gc.ports[1], ch.ports['N'])
    gt.connect(gt.ports[2], gc.ports[2])
    gnd.connect(gnd.ports[1], st.ports[1])
    #######################################################
    # finalize device
    #######################################################
    D = pg.union(D)
    D.flatten(single_layer=layer)
    port_offset = 1
    for i in range(len(halfstages)):
        D.add_port(name=f'clkshift_{port_offset}', port=halfstages[i].ports['shift_clk'])
        D.add_port(name=f'shunt_{port_offset}', port=halfstages[i].ports['shunt'])
        D.add_port(name=f'gnd_{port_offset}', port=halfstages[i].ports['gnd'])
        port_offset += 1
    D.add_port(name='output', port=output_jct.ports[3])
    D.add_port(name='clkro', port=gt.ports[1])
    D.add_port(name=f'gnd_{port_offset}', port=gnd.ports[2])
    return D


def shiftreg_snspd_row(nbn_dev_layer = 1,
                       nbn_pad_layer = 0,
                       via_layer = 2,
                       heater_layer = 3,
                       snspd_w = 0.15,
                       snspd_sq = 10000,
                       heater_w = 0.5,
                       ind_spacings = (6, 3, 6),
                       readout_channel_w = 0.24,
                       readout_gate_w = 0.035,
                       readout_source_sq = 5,
                       readout_drain_sq = 5,
                       readout_gate_sq = 10,
                       ind_sq = 500,
                       dev_outline = 0.2,
                       wire_w = 1,
                       constriction_w = 0.280,
                       routing_w = 5,
                       drain_sq = 5,
                       source_sq = 5,
                       layer = 1):
    """
    shift register with SNSPDs and heaters
    
    parameters:
    snspd_w             - snspd wire width in microns
    snspd_sq            - number of squares in snspd
    heater_w            - width of heater in microns
    ind_spacings        - spacing between ntron column and loop inductor in microns for each stage
    readout_channel_w   - width of readout ntron channel in microns
    readout_gate_w      - width of readout ntron gate in microns
    readout_source_sq   - number of squares for source of readout ntron 
    readout_drain_sq    - number of squares for drain of readout ntron 
    readout_gate_sq     - number of squares for gate of readout ntron 
    ind_sq              - number of squares in loop inductor
    dev_outline         - outline width in microns (all devices are outlined for positive tone)
    wire_w              - width of wire in microns
    constriction_w      - width of channel/gate constriction in microns
    routing_w           - width of routing wires in microns
    drain_sq            - number of squares on drain side
    source_sq           - number of squares on source side
    layer               - gds layer
    """

    D = Device('shiftreg_snspd_row')
    shiftreg = multistage_shiftreg(ind_spacings=ind_spacings, readout_channel_w=readout_channel_w,
                                   readout_gate_w=readout_gate_w, readout_source_sq=readout_source_sq,
                                   readout_drain_sq=readout_drain_sq, readout_gate_sq=readout_gate_sq,
                                   ind_sq=ind_sq, dev_outline=dev_outline, wire_w=wire_w,
                                   constriction_w=constriction_w, routing_w=routing_w, drain_sq=drain_sq,
                                   source_sq=source_sq, layer=nbn_dev_layer)
    sr = D << shiftreg
    snspd_pitch = snspd_w*2-(snspd_w-2*dev_outline)
    snspds = []
    num_snspds = len(ind_spacings)//2
    for i in range(num_snspds):
        S = Device('snspd')
        snspd = S << pg.snspd(wire_width=snspd_w, wire_pitch=snspd_pitch, size=(None, None),
                              num_squares=snspd_sq, layer=nbn_dev_layer)
        bias_taper = S << pg.optimal_step(snspd_w, wire_w, symmetric=True, layer=nbn_dev_layer)
        gnd_taper = S << qg.hyper_taper(2*wire_w, routing_w, snspd_w, layer=nbn_dev_layer)
        bias_tee = S << optimal_tee(width=wire_w, side_length=5*wire_w, layer=nbn_dev_layer)
        bias_taper.connect(bias_taper.ports[1], snspd.ports[1])
        bias_tee.connect(bias_tee.ports[2], bias_taper.ports[2])
        gnd_taper.connect(gnd_taper.ports[1], snspd.ports[2])
        D << S
    for i in range(num_snspds//2):
        # connect snspds on either end of shiftreg
        pass
    if num_snspds % 2 == 1:
        # connect middle snspd
        pass
    return D

def make_device_pair(dev_outline = 0.2,
                     pad_outline = 10,
                     nbn_dev_layer = 1,
                     nbn_pad_layer = 0,
                     via_layer = 2,
                     heater_layer = 3,
                     snspd_w = 0.15,
                     snspd_sq = 1000,
                     heater_width = 0.5,
                     ind_spacings = (6, 3, 6),
                     readout_channel_w = 0.24,
                     readout_gate_w = 0.035,
                     readout_source_sq = 5,
                     readout_drain_sq = 5,
                     readout_gate_sq = 10,
                     ind_sq = 500,
                     wire_w = 1,
                     constriction_w = 0.280,
                     routing_w = 5,
                     drain_sq = 5,
                     source_sq = 5):
    pad_array = qg.pad_array(num=pad_count, size1=(workspace_sidelength, workspace_sidelength),
                             outline=pad_outline, layer=nbn_pad_layer)
    

D = Device("test")
D << shiftreg_snspd_row()
#D << pg.optimal_step(0.1, 1, symmetric=True)
#D << optimal_tee()
#for ind_sq in [200, 500, 1000, 2000]:
##for ind_sq in [500, 1000, 2000]:
#    D << shiftreg(ind_sq=ind_sq)
#D.distribute(direction = 'y', spacing = 10)
#shiftreg_halfstage()
qp(D)

