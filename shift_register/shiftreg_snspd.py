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

import reed as rg

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
    # channel is 1 square
    channel = D << pg.compass(size=(constriction_w, constriction_w), layer=layer)
    source_taper = D << pg.taper(source_sq*wire_w - constriction_w/2, wire_w, constriction_w, layer=layer)
    drain_taper = D << pg.taper(drain_sq*wire_w - constriction_w/2, wire_w, constriction_w, layer=layer)
    gnd_taper = D << qg.hyper_taper(wire_w, routing_w, wire_w, layer=layer)
    source_taper.connect(source_taper.ports[2], channel.ports['S'])
    drain_taper.connect(drain_taper.ports[2], channel.ports['N'])
    gnd_taper.connect(gnd_taper.ports[1], source_taper.ports[1])
    if gate:
        gate_taper = D << pg.taper(wire_w - constriction_w/2, wire_w, constriction_w, layer=layer)
        gate_taper.connect(gate_taper.ports[2], channel.ports['W'])

    #######################################################
    # create tees for shift_clk and shunt
    #######################################################
    tee_jct = rg.optimal_tee(width=wire_w, side_length=5*wire_w, layer=layer)
    bias_jct = D << tee_jct
    shunt_jct = D << tee_jct
    bias_jct.connect(bias_jct.ports[2], drain_taper.ports[1])
    shunt_jct.connect(shunt_jct.ports[1], bias_jct.ports[3])

    #######################################################
    # make loop inductor
    #######################################################
    ind_pitch = wire_w*2-(wire_w-2*dev_outline)
    if ind_sq > 500:
        ind_width = 50*wire_w
    elif ind_sq > 200:
        ind_width = 30*wire_w
    else:
        ind_width = 20*wire_w
    ind_spacer = D << pg.straight(size=(wire_w, (ind_spacing-4*wire_w)), layer=layer)
    ind = D << pg.snspd(wire_width=wire_w, wire_pitch=ind_pitch, size=(ind_width, None),
                        num_squares=ind_sq, layer=layer)
    ind_spacer.connect(ind_spacer.ports[1], bias_jct.ports[3])
    ind.connect(ind.ports[1], ind_spacer.ports[2])

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
            lower = D << pg.connector(midpoint=(gate_taper.ports[1].x-10*wire_w,ind.ports[2].y),width=wire_w)
            r1 = pr.route_smooth(lower.ports[1], gate_taper.ports[1], length1=3*wire_w, length2=3*wire_w, path_type='Z')
            r2 = pg.copy(r1).move([-wire_w,0])
            rcomb = pg.boolean(A=r1, B=r2, operation='or', precision=1e-6, layer=layer)
            D << rcomb
        if snake:
            spacer_height = abs(leftover_height) - min_side_length
            half_snake = rg.optimal_l(width=wire_w, side_length=min_side_length/2, layer=layer)
            upper = D << half_snake
            lower = D << half_snake
            upper.connect(upper.ports[2 if up else 1], gate_taper.ports[1])
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
    D.add_port(name='gnd', port=gnd_taper.ports[2])
    D.add_port(name='shift_clk', port=bias_jct.ports[1])
    D.add_port(name='shunt', port=shunt_jct.ports[3])
    D.add_port(name='output', port=ind.ports[2])
    return D

def multistage_shiftreg(ind_spacings = (6, 3, 6),
                        readout_channel_w = 0.24,
                        readout_gate_w = 0.035,
                        readout_source_sq = 5,
                        readout_drain_sq = 5,
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
    tee_jct = rg.optimal_tee(width=wire_w, side_length=5*wire_w, layer=layer)
    output_jct = D << tee_jct
    output_jct.connect(output_jct.ports[1], halfstages[-1].ports['output'])
    source_taper = D << pg.taper(readout_source_sq*wire_w - readout_channel_w/2, wire_w, readout_channel_w, layer=layer)
    drain_taper = D << pg.taper(readout_drain_sq*wire_w - readout_channel_w/2, wire_w, readout_channel_w, layer=layer)
    # channel is 1 square
    channel = D << pg.compass(size=(readout_channel_w, readout_channel_w), layer=layer)
    gate_taper = D << pg.taper(2*wire_w - readout_channel_w/2 - readout_gate_w/2, wire_w, readout_gate_w, layer=layer)
    gate_choke = D << pg.straight(size=(readout_gate_w,readout_gate_w), layer=layer)
    gnd_taper = D << qg.hyper_taper(wire_w, routing_w, wire_w, layer=layer)
    drain_taper.connect(drain_taper.ports[1], output_jct.ports[2])
    channel.connect(channel.ports['W'], drain_taper.ports[2])
    source_taper.connect(source_taper.ports[2], channel.ports['E'])
    gate_choke.connect(gate_choke.ports[1], channel.ports['N'])
    gate_taper.connect(gate_taper.ports[2], gate_choke.ports[2])
    gnd_taper.connect(gnd_taper.ports[1], source_taper.ports[1])
    gate_ext = D << pg.straight(size=(wire_w,halfstages[-1].ports['shift_clk'].y-gate_taper.ports[1].y), layer=layer)
    gate_ext.connect(gate_ext.ports[1], gate_taper.ports[1])
    output_ext = D << pg.straight(size=(wire_w,halfstages[-1].ports['shift_clk'].y-output_jct.ports[3].y), layer=layer)
    output_ext.connect(output_ext.ports[1], output_jct.ports[3])
    #######################################################
    # finalize device
    #######################################################
    D = pg.union(D)
    D.flatten(single_layer=layer)
    port_offset = 1
    for i in range(len(halfstages)):
        D.add_port(name=f'shiftclk_{port_offset}', port=halfstages[i].ports['shift_clk'])
        D.add_port(name=f'shunt_{port_offset}', port=halfstages[i].ports['shunt'])
        D.add_port(name=f'gnd_{port_offset}', port=halfstages[i].ports['gnd'])
        port_offset += 1
    D.add_port(name='output', port=output_ext.ports[2])
    D.add_port(name='readoutclk', port=gate_ext.ports[2])
    D.add_port(name=f'gnd_{port_offset}', port=gnd_taper.ports[2])
    D = pg.outline(D, distance=dev_outline, open_ports=True, layer=layer)
    return D


def shiftreg_snspd_row(nbn_tron_layer = 2,
                       nbn_snspd_layer = 1,
                       heater_layer = 0,
                       snspd_w = 0.15,
                       snspd_sq = 10000,
                       heater_w = 3,
                       ind_spacings = (10, 3, 10),
                       readout_channel_w = 0.24,
                       readout_gate_w = 0.035,
                       readout_source_sq = 5,
                       readout_drain_sq = 5,
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
    nbn_tron_layer      - gds layer for NbN tron devices
    nbn_snspd_layer     - gds layer for NbN snspds
    heater_layer        - gds layer for Ti/Au heater
    snspd_w             - snspd wire width in microns
    snspd_sq            - number of squares in snspd
    heater_w            - width of heater in microns
    ind_spacings        - spacing between ntron column and loop inductor in microns for each stage
    readout_channel_w   - width of readout ntron channel in microns
    readout_gate_w      - width of readout ntron gate in microns
    readout_source_sq   - number of squares for source of readout ntron 
    readout_drain_sq    - number of squares for drain of readout ntron 
    ind_sq              - number of squares in loop inductor
    dev_outline         - outline width in microns (all devices are outlined for positive tone)
    wire_w              - width of wire in microns
    constriction_w      - width of channel/gate constriction in microns
    routing_w           - width of routing wires in microns
    drain_sq            - number of squares on drain side
    source_sq           - number of squares on source side
    """

    D = Device('shiftreg_snspd_row')
    shiftreg = multistage_shiftreg(ind_spacings=ind_spacings, readout_channel_w=readout_channel_w,
                                   readout_gate_w=readout_gate_w, readout_source_sq=readout_source_sq,
                                   readout_drain_sq=readout_drain_sq, ind_sq=ind_sq,
                                   dev_outline=dev_outline, wire_w=wire_w,
                                   constriction_w=constriction_w, routing_w=routing_w,
                                   drain_sq=drain_sq, source_sq=source_sq, layer=nbn_tron_layer)
    sr = D << shiftreg
    snspd_pitch = snspd_w*2-(snspd_w-2*dev_outline)
    snspd_width = np.ceil(snspd_pitch*(snspd_sq)**0.5)
    snspds = []
    num_snspds = (len(ind_spacings)+1)//2
    for i in range(num_snspds):
        #########################################
        # make the snspd
        #########################################
        S = Device('snspd')
        meander = S << pg.snspd(wire_width=snspd_w, wire_pitch=snspd_pitch, size=(None, None),
                                num_squares=snspd_sq, layer=nbn_snspd_layer)
        bias_taper = S << pg.optimal_step(snspd_w, wire_w, symmetric=True, layer=nbn_snspd_layer)
        gnd_taper = S << qg.hyper_taper(2*wire_w, routing_w, snspd_w, layer=nbn_snspd_layer)
        bias_taper.connect(bias_taper.ports[1], meander.ports[1])
        ext_length = shiftreg.ysize + np.ceil(bias_taper.xsize) - bias_taper.xsize
        taper_ext = S << pg.straight(size=(wire_w, ext_length), layer=nbn_snspd_layer)
        taper_ext.connect(taper_ext.ports[2], bias_taper.ports[2])
        gnd_taper.connect(gnd_taper.ports[1], meander.ports[2])
        #########################################
        # make bumpout in snspd for heater
        #########################################
        curve_sq = 4
        bumpout_curve = rg.optimal_l(width=wire_w, side_length=curve_sq*wire_w, layer=nbn_snspd_layer)
        bc1 = S << bumpout_curve
        bc2 = S << bumpout_curve
        bc2.mirror(p1=[0,0], p2=[0,1])
        bumpout_fill = S << pg.rectangle(size=(heater_w - wire_w, (curve_sq - 1)*wire_w), layer=nbn_snspd_layer)
        bumpout_fill.move((taper_ext.ports[1].x - bumpout_fill.x, taper_ext.ports[1].y - bumpout_fill.y))
        bumpout_fill.move(((9 + drain_sq)*wire_w, (curve_sq - 1)*wire_w/2 + wire_w/2))
        bc1.move((taper_ext.ports[1].x - bc1.xmin, taper_ext.ports[1].y - bc1.ymin))
        bc2.move((taper_ext.ports[1].x - bc2.xmax, taper_ext.ports[1].y - bc2.ymin))
        bc1.move(((9 + drain_sq)*wire_w + (heater_w - wire_w)/2, -wire_w/2))
        bc2.move(((9 + drain_sq)*wire_w - (heater_w - wire_w)/2, -wire_w/2))
        #########################################
        # make the heater/snspd shunt
        #########################################
        # 1 square between snspd and ground; if we make it thick enough it should be ~10-50 Ohms
        res = S << pg.rectangle(size=(heater_w, 2*wire_w + heater_w), layer=heater_layer)
        res.move((taper_ext.ports[1].x - res.x, taper_ext.ports[1].y - res.y))
        res.move(((9 + drain_sq)*wire_w, (curve_sq-1)*wire_w + heater_w/2 + wire_w/2))
        S = pg.union(S, by_layer=True)
        S.add_port(name=1, port=taper_ext.ports[1])
        S.add_port(name=2, port=gnd_taper.ports[2])
        snspd = D << S
        snspd.rotate(270)
        snspd.mirror(p1=[0,0], p2=[0,1])
        snspds.append(snspd)
        dx = shiftreg.ports[f'shiftclk_{2*i+1}'].x + 5*wire_w - snspds[i].ports[1].x
        dy = shiftreg.ports[f'shiftclk_{2*i+1}'].y - snspds[i].ports[1].y
        snspds[i].move((dx,dy))

    return D

def make_device_pair(dev_outline = 0.2,
                     pad_outline = 10,
                     nbn_tron_layer = 2,
                     nbn_snspd_layer = 1,
                     heater_layer = 0,
                     snspd_w = 0.15,
                     snspd_sq = 1000,
                     heater_width = 0.5,
                     ind_spacings = (8, 3, 8),
                     readout_channel_w = 0.24,
                     readout_gate_w = 0.035,
                     readout_source_sq = 5,
                     readout_drain_sq = 5,
                     ind_sq = 500,
                     wire_w = 1,
                     constriction_w = 0.280,
                     routing_w = 5,
                     drain_sq = 5,
                     source_sq = 5):
    # half of devices get onchip bias resistors, other half don't
    pad_array = qg.pad_array(num=pad_count, size1=(workspace_sidelength, workspace_sidelength),
                             outline=pad_outline, layer=nbn_pad_layer)
    

D = Device("test")
#D << pg.optimal_step(0.1, 1, symmetric=True)
#D << rg.optimal_tee()
for ind_sq in [200]:
#for ind_sq in [200, 500, 750, 1000]:
#for ind_sq in [500, 1000, 2000]:
    D << shiftreg_snspd_row(ind_sq=ind_sq)
D.distribute(direction = 'y', spacing = 10)
D.flatten()
D.write_gds('shiftreg_snspd.gds', unit=1e-06, precision=1e-09, auto_rename=True, max_cellname_length=1024, cellname='toplevel')
qp(D)

