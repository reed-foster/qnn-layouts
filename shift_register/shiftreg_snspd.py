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
                       htron_constriction_ratio = 2,
                       constriction_w = 0.280,
                       routing_w = 5,
                       drain_sq = 10,
                       source_sq = 5,
                       gate = True,
                       layer = 1):
    """
    shift register halfstage

    vertical column with ntron, optional input, shift_clk, and shunt followed by loop inductor
    parameters:
    ind_sq                      - number of squares in loop inductor
    ind_spacing                 - spacing between ntron column and loop inductor in microns (prevent heating of loop)
    dev_outline                 - outline width in microns (all devices are outlined for positive tone)
    wire_w                      - width of wire in microns
    htron_constriction_ratio    - ratio of htron constriction to ntron constriction (should be > 1)
    constriction_w              - width of channel/gate constriction in microns
    routing_w                   - width of routing wires in microns
    drain_sq                    - number of squares on drain side
    source_sq                   - number of squares on source side
    gate                        - True if gate should be included, False otherwise
    layer                       - gds layer
    """

    D = Device("sr_halfstage")

    htron_channel_w = htron_constriction_ratio*constriction_w
    if not gate:
        constriction_w = wire_w
    #######################################################
    # create ntron
    #######################################################
    # channel is 1 square
    channel = D << pg.compass(size=(constriction_w, constriction_w), layer=layer)
    source_taper = D << pg.taper(source_sq*wire_w - constriction_w/2, wire_w, constriction_w, layer=layer)
    drain_taper = D << pg.taper(drain_sq/2*wire_w - constriction_w/2, wire_w, constriction_w, layer=layer)
    gnd_taper = D << qg.hyper_taper(length=wire_w, wide_section=routing_w, narrow_section=wire_w, layer=layer)
    source_taper.connect(source_taper.ports[2], channel.ports['S'])
    drain_taper.connect(drain_taper.ports[2], channel.ports['N'])
    gnd_taper.connect(gnd_taper.ports[1], source_taper.ports[1])
    if gate:
        gate_taper = D << pg.taper(wire_w - constriction_w/2, wire_w, constriction_w, layer=layer)
        gate_taper.connect(gate_taper.ports[2], channel.ports['W'])
    
    #######################################################
    # create htron taper
    #######################################################
    htron_spacer = D << pg.straight(size=(wire_w, drain_sq/4*wire_w), layer=layer)
    htron_channel = D << pg.compass(size=(htron_channel_w, htron_channel_w), layer=layer)
    htron_source_taper = D << pg.taper(drain_sq/8*wire_w - htron_channel_w/2, wire_w, htron_channel_w, layer=layer)
    htron_drain_taper = D << pg.taper(drain_sq/8*wire_w - htron_channel_w/2, wire_w, htron_channel_w, layer=layer)
    htron_spacer.connect(htron_spacer.ports[1], drain_taper.ports[1])
    htron_source_taper.connect(htron_source_taper.ports[1], htron_spacer.ports[2])
    htron_channel.connect(htron_channel.ports['S'], htron_source_taper.ports[2])
    htron_drain_taper.connect(htron_drain_taper.ports[2], htron_channel.ports['N'])

    #######################################################
    # create tees for shift_clk and shunt
    #######################################################
    tee_jct = rg.optimal_tee(width=wire_w, side_length=5*wire_w, layer=layer)
    bias_jct = D << tee_jct
    shunt_jct = D << tee_jct
    bias_jct.connect(bias_jct.ports[2], htron_drain_taper.ports[1])
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
        pre_spacer = D << pg.straight(size=(wire_w, 5*wire_w), layer=layer)
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
            lower = D << pg.connector(midpoint=(gate_taper.ports[1].x-7*wire_w,ind.ports[2].y),width=wire_w)
            r1 = pr.route_smooth(lower.ports[1], gate_taper.ports[1], length1=2*wire_w, length2=2*wire_w, path_type='Z')
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
            pre_spacer.connect(pre_spacer.ports[1], lower.ports[2 if up else 1])
        else:
            pre_spacer.connect(pre_spacer.ports[1], lower.ports[2])

    #######################################################
    # finalize device
    #######################################################
    D = pg.union(D)
    D.flatten(single_layer=layer)
    if gate:
        D.add_port(name='input', port=pre_spacer.ports[2])
    D.add_port(name='gnd', port=gnd_taper.ports[2])
    D.add_port(name='shift_clk', port=bias_jct.ports[1])
    D.add_port(name='shunt', port=shunt_jct.ports[3])
    D.add_port(name='output', port=ind.ports[2])
    return D

def multistage_shiftreg(ind_spacings = (10, 5, 10),
                        readout_channel_w = 0.24,
                        readout_gate_w = 0.035,
                        readout_source_sq = 5,
                        readout_drain_sq = 5,
                        ind_sq = 500,
                        dev_outline = 0.2,
                        wire_w = 1,
                        htron_constriction_ratio = 2,
                        constriction_w = 0.280,
                        routing_w = 5,
                        drain_sq = 10,
                        source_sq = 5,
                        layer = 1):
    """
    shift register
    
    cascaded half stages, each with individually settable inductor spacing
    parameters:
    ind_spacings                - spacing between ntron column and loop inductor in microns for each stage
    readout_channel_w           - width of readout ntron channel in microns
    readout_gate_w              - width of readout ntron gate in microns
    readout_source_sq           - number of squares for source of readout ntron 
    readout_drain_sq            - number of squares for drain of readout ntron 
    ind_sq                      - number of squares in loop inductor
    dev_outline                 - outline width in microns (all devices are outlined for positive tone)
    wire_w                      - width of wire in microns
    htron_constriction_ratio    - ratio of htron constriction to ntron constriction (should be > 1)
    constriction_w              - width of channel/gate constriction in microns
    routing_w                   - width of routing wires in microns
    drain_sq                    - number of squares on drain side
    source_sq                   - number of squares on source side
    layer                       - gds layer
    """

    D = Device('multistage_shiftreg')

    #######################################################
    # make halfstages
    #######################################################
    halfstages = []
    for i, ind_spacing in enumerate(ind_spacings):
        hcr = htron_constriction_ratio if (i % 2 == 0) else wire_w/constriction_w
        halfstages.append(D << shiftreg_halfstage(ind_sq=ind_sq, ind_spacing=ind_spacing,
                                                  dev_outline=dev_outline, wire_w=wire_w,
                                                  htron_constriction_ratio=hcr, 
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
    gnd_taper = D << qg.hyper_taper(length=wire_w, wide_section=routing_w, narrow_section=wire_w, layer=layer)
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
    for i in range(len(halfstages)):
        D.add_port(name=3*i, port=halfstages[i].ports['shift_clk'])
        D.add_port(name=3*i+1, port=halfstages[i].ports['shunt'])
        D.add_port(name=3*i+2, port=halfstages[i].ports['gnd'])
    offset = 3*len(halfstages)
    D.add_port(name=offset, port=output_ext.ports[2])
    D.add_port(name=offset+1, port=gate_ext.ports[2])
    D.add_port(name=offset+2, port=gnd_taper.ports[2])
    D = pg.outline(D, distance=dev_outline, open_ports=True, layer=layer)
    return D


def shiftreg_snspd_row(nbn_tron_layer = 2,
                       nbn_snspd_layer = 1,
                       heater_layer = 0,
                       snspd = True,
                       snspd_w = 0.3,
                       snspd_sq = 10000,
                       heater_w = 3,
                       ind_spacings = (10, 5, 10),
                       readout_channel_w = 0.24,
                       readout_gate_w = 0.035,
                       readout_source_sq = 5,
                       readout_drain_sq = 5,
                       ind_sq = 500,
                       dev_outline = 0.2,
                       wire_w = 1,
                       htron_constriction_ratio = 2,
                       constriction_w = 0.280,
                       routing_w = 5,
                       drain_sq = 10,
                       source_sq = 5):
    """
    shift register with SNSPDs and heaters
    
    parameters:
    nbn_tron_layer              - gds layer for NbN tron devices
    nbn_snspd_layer             - gds layer for NbN snspds
    heater_layer                - gds layer for Ti/Au heater
    snspd                       - True if snspd meander should be fabbed, False if just the heater
    snspd_w                     - snspd wire width in microns
    snspd_sq                    - number of squares in snspd
    heater_w                    - width of heater in microns
    ind_spacings                - spacing between ntron column and loop inductor in microns for each stage
    readout_channel_w           - width of readout ntron channel in microns
    readout_gate_w              - width of readout ntron gate in microns
    readout_source_sq           - number of squares for source of readout ntron 
    readout_drain_sq            - number of squares for drain of readout ntron 
    ind_sq                      - number of squares in loop inductor
    dev_outline                 - outline width in microns (all devices are outlined for positive tone)
    wire_w                      - width of wire in microns
    htron_constriction_ratio    - ratio of htron constriction to ntron constriction (should be > 1)
    constriction_w              - width of channel/gate constriction in microns
    routing_w                   - width of routing wires in microns
    drain_sq                    - number of squares on drain side
    source_sq                   - number of squares on source side
    """

    D = Device('shiftreg_snspd_row')
    shiftreg = multistage_shiftreg(ind_spacings=ind_spacings, readout_channel_w=readout_channel_w,
                                   readout_gate_w=readout_gate_w, readout_source_sq=readout_source_sq,
                                   readout_drain_sq=readout_drain_sq, ind_sq=ind_sq,
                                   dev_outline=dev_outline, wire_w=wire_w,
                                   htron_constriction_ratio=htron_constriction_ratio,
                                   constriction_w=constriction_w, routing_w=routing_w,
                                   drain_sq=drain_sq, source_sq=source_sq, layer=nbn_tron_layer)
    sr = D << shiftreg
    snspd_pitch = snspd_w*2 - (snspd_w - 2*dev_outline)
    snspd_width = np.ceil(snspd_pitch*(snspd_sq**0.5))
    snspds = []
    num_snspds = (len(ind_spacings) + 1)//2
    for i in range(num_snspds):
        S = Device('snspd')
        # number of squares for curve to resistor taper
        curve_sq = 4
        #########################################
        # make bumpout in snspd bias for heater
        #########################################
        if snspd:
            bumpout_curve = S << rg.optimal_tee(width=(wire_w, heater_w + wire_w),
                                                side_length=curve_sq*wire_w, layer=nbn_snspd_layer)
        else:
            bumpout_curve = S << rg.optimal_l(width=(wire_w, heater_w + wire_w),
                                              side_length=curve_sq*wire_w, layer=nbn_snspd_layer)
        straight_length = (9 + drain_sq/8)*wire_w - (curve_sq - 1)*wire_w - (heater_w + wire_w)/2
        bias_straight = S << pg.straight(size=(wire_w, straight_length), layer=nbn_snspd_layer)
        #########################################
        # make the snspd
        #########################################
        if snspd:
            M = Device('meander')
            m = M << pg.snspd(wire_width=snspd_w, wire_pitch=snspd_pitch, size=(None, None),
                              num_squares=snspd_sq, layer=nbn_snspd_layer)
            if nbn_snspd_layer < nbn_tron_layer:
                cutout = M << pg.rectangle(size=(m.xsize + 2*wire_w, m.ysize + 2*wire_w), layer=nbn_tron_layer)
                cutout.move((m.x - cutout.x, m.y - cutout.y))
            M.add_port(name=1, port=m.ports[1])
            M.add_port(name=2, port=m.ports[2])
            meander = S << M
            bias_taper = S << pg.optimal_step(snspd_w, wire_w, symmetric=True, layer=nbn_snspd_layer)
            gnd_taper = S << qg.hyper_taper(length=2*wire_w, wide_section=routing_w, narrow_section=snspd_w, layer=nbn_snspd_layer)
            bias_taper.connect(bias_taper.ports[1], meander.ports[1])
            ext_length = shiftreg.ysize - 2*dev_outline + np.ceil(bias_taper.xsize) - bias_taper.xsize
            ext_length -= straight_length + 2*(curve_sq - 1)*wire_w + heater_w + wire_w
            taper_ext = S << pg.straight(size=(wire_w, ext_length), layer=nbn_snspd_layer)
            taper_ext.connect(taper_ext.ports[2], bias_taper.ports[2])
            gnd_taper.connect(gnd_taper.ports[1], meander.ports[2])
            bumpout_curve.connect(bumpout_curve.ports[1], taper_ext.ports[1])
            bias_straight.connect(bias_straight.ports[1], bumpout_curve.ports[2])
        else:
            bias_straight.rotate(270)
            bumpout_curve.mirror((0,0), (0,1))
            bumpout_curve.connect(bumpout_curve.ports[1], bias_straight.ports[1])
        #########################################
        # make the heater/snspd shunt
        #########################################
        # 1 square between snspd and ground; if we make it thick enough it should be ~10-50 Ohms
        res = S << pg.rectangle(size=(heater_w, 2*wire_w + heater_w), layer=heater_layer)
        if snspd:
            res.move((bumpout_curve.ports[3].x - res.x, bumpout_curve.ports[3].y - res.ymin))
        else:
            res.move((bumpout_curve.ports[2].x - res.x, bumpout_curve.ports[2].y - res.ymin))
        res.move((0, -wire_w))
        #########################################
        # make the ground return for the shunt resistor
        #########################################
        # straight from resistor to connect to smaller turn
        res_gnd = S << pg.straight(size=(heater_w + wire_w, 5*wire_w), layer=nbn_snspd_layer)
        res_gnd.move((res.x - res_gnd.ports[2].x, res.ymax - res_gnd.ports[2].y))
        res_gnd.move((0, -wire_w))
        res_gnd_turn = S << rg.optimal_l(width=(heater_w + wire_w, wire_w), side_length=4*wire_w, layer=nbn_snspd_layer)
        res_gnd_turn.mirror((0,0),(0,1))
        res_gnd_turn.connect(res_gnd_turn.ports[1], res_gnd.ports[1])
        # straight long enough to go past the gate bias line on the ntron layer
        res_gnd_ext_length = shiftreg.ysize - 2*dev_outline - straight_length - 2*(curve_sq - 1)*wire_w - heater_w - wire_w
        res_gnd_ext = S << pg.straight(size=(wire_w, res_gnd_ext_length), layer=nbn_snspd_layer)
        res_gnd_ext.connect(res_gnd_ext.ports[1], res_gnd_turn.ports[2])
        if snspd:
            # step up to routing_w
            res_gnd_step = S << pg.optimal_step(wire_w, routing_w, symmetric=True, layer=nbn_snspd_layer)
            res_gnd_step.connect(res_gnd_step.ports[1], res_gnd_ext.ports[2])
            res_gnd_conn = S << pg.straight(size=(routing_w, gnd_taper.ports[2].x - res_gnd_step.ports[2].x - routing_w),
                                            layer=nbn_snspd_layer)
            res_gnd_conn.connect(res_gnd_conn.ports[1], res_gnd_step.ports[2])
        if snspd:
            # hyper taper to ground
            res_gnd_taper = S << qg.hyper_taper(length=routing_w, wide_section=2*routing_w,
                                                narrow_section=routing_w, layer=nbn_snspd_layer)
            res_gnd_taper.connect(res_gnd_taper.ports[1], res_gnd_conn.ports[2])
        else:
            # hyper taper to ground
            res_gnd_taper = S << qg.hyper_taper(length=routing_w, wide_section=2*routing_w,
                                                narrow_section=wire_w, layer=nbn_snspd_layer)
            res_gnd_taper.connect(res_gnd_taper.ports[1], res_gnd_ext.ports[2])
        S = pg.union(S, by_layer=True)
        S.add_port(name=1, port=bias_straight.ports[2])
        if snspd:
            S.add_port(name=2, port=gnd_taper.ports[2])
        S.add_port(name=0, port=res_gnd_taper.ports[2])
        snspd_dev = D << S
        snspd_dev.rotate(270)
        snspd_dev.mirror((0,0), (0,1))
        snspds.append(snspd_dev)
        dx = shiftreg.ports[6*i].x + (1+curve_sq)*wire_w - snspds[i].ports[1].x
        dy = shiftreg.ports[6*i].y - snspds[i].ports[1].y
        snspds[i].move((dx,dy))
    # connect ground for snspds
    left_gnd = snspds[0].ports[0]
    right_gnd = snspds[-1].ports[2 if snspd else 0]
    snspd_gnd_pour_w = right_gnd.endpoints[0][0] - left_gnd.endpoints[1][0]
    snspd_gnd_pour_h = 2*routing_w
    snspd_gnd_pour = D << pg.straight(size=(snspd_gnd_pour_h, snspd_gnd_pour_w), layer=nbn_snspd_layer)
    snspd_gnd_pour.rotate(90)
    snspd_gnd_pour.move(left_gnd.endpoints[1] - np.array([snspd_gnd_pour.xmin, snspd_gnd_pour.ymax - 1e-4]))
    D = pg.union(D, by_layer=True)
    D.add_port(name=0, port=snspd_gnd_pour.ports[1])
    for i in range(num_snspds):
        D.add_port(name=5*i+1, port=snspds[i].ports[1])
        for pid in range(2):
            D.add_port(name=5*i+pid+2, port=shiftreg.ports[6*i+pid])
            D.add_port(name=5*i+pid+4, port=shiftreg.ports[6*i+3+pid])
    return D

def make_device_pair(dev_outline = 0.2,
                     pad_outline = 10,
                     nbn_tron_pad_layer = 4,
                     nbn_snspd_pad_layer = 2,
                     nbn_tron_layer = 3,
                     nbn_snspd_layer = 1,
                     heater_layer = 0,
                     snspd_w = 0.3,
                     snspd_sq = 10000,
                     heater_w = 3,
                     ind_spacings = (10, 5, 10),
                     readout_channel_w = 0.24,
                     readout_gate_w = 0.035,
                     readout_source_sq = 5,
                     readout_drain_sq = 5,
                     ind_sq = 500,
                     wire_w = 1,
                     htron_constriction_ratio = 2,
                     constriction_w = 0.280,
                     routing_w = 5,
                     drain_sq = 10,
                     source_sq = 5):
    D = Device('shiftreg_experiment')
    # half of devices get snspd inputs, the other half just get a wire
    E = Device('experiment')
    shiftregs = []
    port_offset = 0
    for snspd in (True, False):
        shiftreg = E << shiftreg_snspd_row(nbn_tron_layer=nbn_tron_layer, nbn_snspd_layer=nbn_snspd_layer,
                                           heater_layer=heater_layer, snspd=snspd, snspd_w=snspd_w,
                                           snspd_sq=snspd_sq, heater_w=heater_w, ind_spacings=ind_spacings,
                                           readout_channel_w=readout_channel_w, readout_gate_w=readout_gate_w,
                                           readout_source_sq=readout_source_sq, readout_drain_sq=readout_drain_sq,
                                           ind_sq=ind_sq, dev_outline=dev_outline, wire_w=wire_w,
                                           htron_constriction_ratio=htron_constriction_ratio,
                                           constriction_w=constriction_w, routing_w=routing_w, drain_sq=drain_sq,
                                           source_sq=source_sq)
        if not snspd:
            shiftreg.rotate(180)
            shiftreg.move((shiftregs[0].x - shiftreg.x, shiftregs[0].ymin - shiftreg.ymax + shiftreg.ports[0].width))
        shiftregs.append(shiftreg)
        # add ports
        for pid, port in shiftreg.ports.items():
            if not snspd and pid == 0:
                continue
            E.add_port(name=port_offset+pid, port=port)
        port_offset += len(shiftreg.ports.items()) - 1
    E = pg.union(E, by_layer=True)
    E.move((-E.x, -E.y))
    D << E
    snspd_pad_count = (len(ind_spacings) + 1)//2 + 1
    ntron_pad_count = (len(ind_spacings) + 1)*2
    pad_count = (snspd_pad_count + ntron_pad_count)*2 - 1
    workspace_size = 1.5*shiftregs[0].xsize
    pad_array = rg.pad_array(num_pads=pad_count, workspace_size=workspace_size,
                             pad_layers=tuple(nbn_snspd_pad_layer for i in range(pad_count)),
                             outline=pad_outline,
                             pad_size=(200,250),
                             pos_tone= {nbn_snspd_pad_layer: False, nbn_tron_pad_layer: True})
    pa = D << pad_array
    # route from devices to pad array
    return D
    

D = Device("test")
D << make_device_pair(ind_spacings=(10, 5, 10), ind_sq=200)
qp(D)
exit()
#D << pg.optimal_step(0.1, 1, symmetric=True)
#D << rg.optimal_tee()
for ind_sq in [200]:
#for ind_sq in [200, 500, 750, 1000]:
#for ind_sq in [500, 1000, 2000]:
    D << shiftreg_snspd_row(ind_sq=ind_sq, ind_spacings=(20, 5, 20, 5, 20), wire_w=1, snspd=True)
    D << shiftreg_snspd_row(ind_sq=ind_sq, ind_spacings=(20, 5, 20, 5, 20), wire_w=1, snspd=False)
    #D << multistage_shiftreg()
#D.distribute(direction = 'y', spacing = 10)
#D.write_gds('shiftreg_snspd.gds', unit=1e-06, precision=1e-09, auto_rename=True, max_cellname_length=1024, cellname='toplevel')
qp(D)

