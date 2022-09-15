# -*- coding: utf-8 -*-
"""
Created on Wed Aug 31 14:00:00 2022

@author: reedf

"""

import numpy as np
from phidl import Device
from phidl import Path
import phidl.geometry as pg
import phidl.routing as pr
import phidl.path as pp
from phidl import quickplot as qp
from phidl import set_quickplot_options
#set_quickplot_options(blocking=True, show_ports=True)
#set_quickplot_options(blocking=True, show_ports=False)
set_quickplot_options(blocking=False, show_ports=True, new_window=True)

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
    bias_jct.connect(bias_jct.ports[2], htron_drain_taper.ports[1])

    #######################################################
    # make loop inductor
    #######################################################
    ind_pitch = wire_w*2-(wire_w-2*dev_outline)
    ind_width = 50*wire_w if ind_sq > 500 else (30*wire_w if ind_sq > 200 else 20*wire_w)
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
            lower = D << pg.connector(midpoint=(gate_taper.ports[1].x-12*wire_w,ind.ports[2].y),width=wire_w)
            r1 = pr.route_smooth(lower.ports[1], gate_taper.ports[1], length1=2*wire_w, length2=2*wire_w, path_type='Z')
            r2 = pg.copy(r1).move([-wire_w,0])
            rcomb = pg.boolean(A=r1, B=r2, operation='or', precision=1e-6, layer=layer)
            D << rcomb
        if snake:
            pre_spacer = D << pg.straight(size=(wire_w, 5*wire_w), layer=layer)
            spacer_height = abs(leftover_height) - min_side_length
            half_snake = rg.optimal_l(width=wire_w, side_length=min_side_length/2, layer=layer)
            upper = D << half_snake
            lower = D << half_snake
            upper.connect(upper.ports[2 if up else 1], gate_taper.ports[1])
            snake_sp = D << pg.straight(size=(wire_w, spacer_height), layer=layer)
            snake_sp.connect(snake_sp.ports[1], upper.ports[1 if up else 2])
            lower.connect(lower.ports[1 if up else 2], snake_sp.ports[2])
            pre_spacer.connect(pre_spacer.ports[1], lower.ports[2 if up else 1])

    #######################################################
    # finalize device
    #######################################################
    D = pg.union(D)
    D.flatten(single_layer=layer)
    if gate:
        D.add_port(name='input', port=pre_spacer.ports[2] if snake else lower.ports[2])
    D.add_port(name='gnd', port=gnd_taper.ports[2])
    D.add_port(name='shift_clk', port=bias_jct.ports[1])
    D.add_port(name='output', port=ind.ports[2])
    D.add_port(name='htron_ch', midpoint=htron_channel.center, width=htron_channel_w, orientation=180)
    return D

def multistage_shiftreg(stage_count = 3,
                        ind_spacing = 10,
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
    stage_count                 - number of stages
    ind_spacing                 - spacing between ntron column and loop inductor in microns for each stage
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
    for i in range(stage_count):
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
    gate_ext = D << rg.optimal_l(width=wire_w, side_length=5*wire_w, layer=layer)
    gate_ext.connect(gate_ext.ports[2], gate_taper.ports[1])
    output_ext = D << rg.optimal_l(width=wire_w, side_length=15*wire_w, layer=layer)
    output_ext.connect(output_ext.ports[2], output_jct.ports[3])
    #######################################################
    # finalize device
    #######################################################
    D = pg.union(D)
    D.flatten(single_layer=layer)
    for i in range(len(halfstages)):
        D.add_port(name=3*i, port=halfstages[i].ports['shift_clk'])
        D.add_port(name=3*i+1, port=halfstages[i].ports['gnd'])
    offset = 3*len(halfstages)
    D.add_port(name=offset, port=output_ext.ports[1])
    D.add_port(name=offset+1, port=gate_ext.ports[1])
    D.add_port(name=offset+2, port=gnd_taper.ports[2])
    D = pg.outline(D, distance=dev_outline, open_ports=True, layer=layer)
    # add this afterwards, otherwise it will disrupt the outline
    for i in range(len(halfstages)):
        if i % 2 == 0:
            D.add_port(name=3*i+2, port=halfstages[i].ports['htron_ch'])
    return D

def shiftreg_snspd_row(nbn_layer = 0,
                       via_layer = 1,
                       heater_layer = 2,
                       onchip_bias = True,
                       snspd = False,
                       snspd_w = 0.3,
                       snspd_ff = 0.3,
                       snspd_sq = 10000,
                       heater_w = 3,
                       stage_count = 3,
                       ind_spacing = 10,
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
    nbn_layer                   - gds layer for NbN devices (pos. tone, NbN etch)
    via_layer                   - gds layer for SiO2 etch pattern (pos. tone, SiO2 etch)
    heater_layer                - gds layer for Ti/Au heater (pos. tone, liftoff)
    onchip_bias                 - True if heater layer should be used for making onchip bias resistors
    snspd                       - True if snspd meander should be fabbed, False if just the heater
    snspd_w                     - snspd wire width in microns
    snspd_ff                    - snspd fill factor (as ratio between 0 and 1)
    snspd_sq                    - number of squares in snspd
    heater_w                    - width of heater in microns
    stage_count                 - number of shift register stages
    ind_spacing                 - spacing between ntron column and loop inductor in microns for each stage
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
    # check inputs
    if onchip_bias and snspd:
        raise ValueError('onchip bias not supported for SNSPDs yet')

    shiftreg = multistage_shiftreg(stage_count=stage_count, ind_spacing=ind_spacing,
                                   readout_channel_w=readout_channel_w, readout_gate_w=readout_gate_w,
                                   readout_source_sq=readout_source_sq, readout_drain_sq=readout_drain_sq,
                                   ind_sq=ind_sq, dev_outline=dev_outline, wire_w=wire_w,
                                   htron_constriction_ratio=htron_constriction_ratio,
                                   constriction_w=constriction_w, routing_w=routing_w,
                                   drain_sq=drain_sq, source_sq=source_sq, layer=nbn_layer)
    sr = D << shiftreg
    snspd_pitch = snspd_w/snspd_ff
    snspd_width = np.ceil(snspd_pitch*(snspd_sq**0.5))
    snspds = []
    num_snspds = (stage_count + 1)//2
    for i in range(num_snspds):
        S = Device('snspd')
        # number of squares for curve to resistor taper
        curve_sq = 5
        ########################################################
        # make bumpout in snspd bias for heater/shunt resistor
        ########################################################
        if not snspd and i == 0 and not onchip_bias:
            bumpout_curve = S << pg.straight(size=(routing_w, routing_w), layer=nbn_layer)
            bumpout_curve.connect(bumpout_curve.ports[2], shiftreg.ports[6*i + 2])
            bumpout_curve.move((-heater_w/2, 0))
        else:
            if snspd:
                bumpout_curve = S << rg.optimal_l(width=(wire_w, heater_w + wire_w),
                                                  side_length=curve_sq*wire_w, layer=nbn_layer)
            else:
                # make the curve a bit wider if no snspd in case there are constrictions or htron
                # just needs more heater current than can be supplied by the thin wire
                bumpout_curve = S << rg.optimal_l(width=(3*wire_w, heater_w + wire_w),
                                                  side_length=2*curve_sq*wire_w, layer=nbn_layer)
            bumpout_curve.connect(bumpout_curve.ports[2], shiftreg.ports[6*i + 2])
            bumpout_curve.move((-heater_w/2, 0))
        # taper for going from wire_w to routing_w
        wire_taper = pg.optimal_step(start_width=wire_w, end_width=routing_w, symmetric=True, layer=nbn_layer)
        if snspd:
            if i > 0:
                hairpin = S << pg.optimal_hairpin(width=wire_w, pitch=routing_w,
                                                  length=(6 + drain_sq/4)*wire_w, turn_ratio=2, layer=nbn_layer)
            else:
                hairpin = S << rg.optimal_l(width=(wire_w, routing_w),
                                            side_length=(7 + drain_sq/4)*wire_w - routing_w, layer=nbn_layer)
                # optimal_l has port 1 facing up and port 2 facing right
                # we want this opimal_l to be facing the opposite direction of the bumpout_curve
                hairpin.mirror((0,0), (0,1))
            connector = S << pg.connector(width=wire_w)
            connector.connect(connector.ports[1], bumpout_curve.ports[1])
            hairpin.connect(hairpin.ports[1], connector.ports[1])
            hairpin.move((0, bumpout_curve.ymin - hairpin.ymin))
            wide_l = rg.optimal_l(width=wire_w, side_length=2*curve_sq*wire_w, layer=nbn_layer)
            thin_l = rg.optimal_l(width=wire_w, side_length=curve_sq*wire_w, layer=nbn_layer)
            turn_1 = S << wide_l
            turn_2 = S << thin_l
            straight_len = shiftreg.ports[6].x - shiftreg.ports[3].x - 33*wire_w
            straight_t = pg.straight(size=(wire_w, straight_len), layer=nbn_layer)
            straight_1 = S << straight_t
            turn_1.connect(turn_1.ports[1], hairpin.ports[1])
            straight_1.connect(straight_1.ports[1], turn_1.ports[2])
            turn_2.connect(turn_2.ports[2], straight_1.ports[2])
            if i > 0:
                turn_3 = S << thin_l
                straight_2 = S << straight_t
                turn_4 = S << wide_l
                turn_3.connect(turn_3.ports[1], hairpin.ports[2])
                straight_2.connect(straight_2.ports[1], turn_3.ports[2])
                turn_4.connect(turn_4.ports[2], straight_2.ports[2])
            step = S << pg.optimal_step(start_width=snspd_w, end_width=wire_w, symmetric=True, layer=nbn_layer)
            meander = S << pg.snspd(wire_width=snspd_w, wire_pitch=snspd_pitch, size=(None,None),
                                    num_squares=snspd_sq, layer=nbn_layer)
            gnd_taper = S << qg.hyper_taper(length=routing_w, wide_section=routing_w, narrow_section=snspd_w,
                                            layer=nbn_layer)
            step.connect(step.ports[2], turn_2.ports[1])
            meander.connect(meander.ports[2], step.ports[1])
            gnd_taper.connect(gnd_taper.ports[1], meander.ports[1])
            if i > 0:
                snspd_taper = S << wire_taper
                snspd_taper.connect(snspd_taper.ports[1], turn_4.ports[1])
            shiftreg_clk_blocked = shiftreg.ports[6*i].x < gnd_taper.ports[2].x + 1.5*routing_w + wire_w
            if shiftreg_clk_blocked:
                # add snake to shiftreg clock if it's blocked by SNSPD
                turn_5 = S << thin_l
                turn_6 = S << thin_l
                straight_len = gnd_taper.ports[2].x + 1.5*routing_w + wire_w - shiftreg.ports[6*i].x - 2*curve_sq*wire_w
                straight_3 = S << pg.straight(size=(wire_w, max(straight_len, 0)), layer=nbn_layer)
                turn_5.connect(turn_5.ports[2], shiftreg.ports[6*i])
                straight_3.connect(straight_3.ports[1], turn_5.ports[1])
                turn_6.connect(turn_6.ports[1], straight_3.ports[2])
            # add taper for shiftreg clock
            clk_taper_1 = S << wire_taper
            clk_taper_2 = S << wire_taper
            clk_taper_1.connect(clk_taper_1.ports[1], turn_6.ports[2] if shiftreg_clk_blocked else shiftreg.ports[6*i])
            clk_taper_2.connect(clk_taper_2.ports[1], shiftreg.ports[6*i + 3])
        else:
            if i > 0 or onchip_bias:
                if snspd:
                    snspd_taper = S << wire_taper
                    snspd_taper.connect(snspd_taper.ports[1], bumpout_curve.ports[1])
                else:
                    if i >= (num_snspds + 1) // 2:
                        # make a turn
                        taper_turn = S << rg.optimal_l(width=3*wire_w, side_length=12*wire_w, layer=nbn_layer)
                        snspd_taper = S << rg.optimal_l(width=(3*wire_w, routing_w), side_length=4*routing_w,
                                                        layer=nbn_layer).mirror((0,0), (0,1))
                        taper_turn.connect(taper_turn.ports[1], bumpout_curve.ports[1])
                        snspd_taper.connect(snspd_taper.ports[1], taper_turn.ports[2])
                    else:
                        snspd_taper = S << pg.optimal_step(start_width=3*wire_w, end_width=routing_w,
                                                           symmetric=True, layer=nbn_layer)
                        snspd_taper.connect(snspd_taper.ports[1], bumpout_curve.ports[1])
            # add taper for shiftreg clock
            clk_taper_1 = S << wire_taper
            clk_taper_2 = S << wire_taper
            clk_taper_1.connect(clk_taper_1.ports[1], shiftreg.ports[6*i])
            clk_taper_2.connect(clk_taper_2.ports[1], shiftreg.ports[6*i + 3])
        ##################################################################################
        # do outline for positive tone and fix snspd cutout to meet snspd_ff
        ##################################################################################
        S = pg.union(S, by_layer=True)
        # add ports first
        if snspd:
            S.add_port(name=0, port=snspd_taper.ports[2] if i > 0 else hairpin.ports[2])
            S.add_port(name=1, port=gnd_taper.ports[2])
            S.add_port(name=2, port=clk_taper_1.ports[2])
            S.add_port(name=3, port=turn_5.ports[2] if shiftreg_clk_blocked else clk_taper_1.ports[1])
            S.add_port(name=4, port=clk_taper_2.ports[2])
            S.add_port(name=5, port=clk_taper_2.ports[1])
        else:
            if not onchip_bias:
                S.add_port(name=0, port=snspd_taper.ports[2] if i > 0 else bumpout_curve.ports[1])
            S.add_port(name=2, port=clk_taper_1.ports[2])
            S.add_port(name=3, port=clk_taper_1.ports[1])
            S.add_port(name=4, port=clk_taper_2.ports[2])
            S.add_port(name=5, port=clk_taper_2.ports[1])
        # do outline
        S = pg.outline(S, distance=dev_outline, open_ports=True, layer=nbn_layer)
        if onchip_bias:
            S.add_port(name=0, port=snspd_taper.ports[2])
        if snspd:
            # add pour over snspd and re-add meander
            pour = pg.rectangle(size=(meander.xsize + 2*dev_outline, meander.ysize + 2*dev_outline),
                                layer=nbn_layer)
            pour.move((meander.x - pour.x, meander.y - pour.y))
            M = pg.snspd(wire_width=snspd_w, wire_pitch=snspd_pitch, size=(None,None),
                         num_squares=snspd_sq, layer=nbn_layer).rotate(90)
            M.move((meander.x - M.x, meander.y - M.y))
            straight = pg.rectangle(size=(snspd_w, dev_outline))
            s1 = M << straight
            s2 = M << straight
            s1.move((M.ports[1].x - s1.x, M.ports[1].y - s1.ymax))
            s2.move((M.ports[2].x - s2.x, M.ports[2].y - s2.ymin))
            S << pg.boolean(A=pour, B=M, operation='not', precision=1e-6, num_divisions=(1,1), layer=nbn_layer)
        #########################################
        # make the heater/snspd shunt and vias
        #########################################
        # 1 square between snspd and ground; if we make it thick enough it should be ~10-50 Ohms
        res = S << rg.resistor_with_vias(via_layer=via_layer, res_layer=heater_layer, res_w=heater_w,
                                         res_sq=1, via_max_w=(bumpout_curve.ports[2].width - 2, None))
        res.connect(res.ports[1], bumpout_curve.ports[2])
        res.move((-res.ports[1].width - 0.5, 0))
        snspds.append(D << S)
    # add readout clock taper
    clk_ro_taper = D << pg.outline(pg.optimal_step(start_width=wire_w, end_width=routing_w, symmetric=True),
                                   distance=dev_outline, open_ports=True, layer=nbn_layer)
    clk_ro_taper.connect(clk_ro_taper.ports[1], shiftreg.ports[6*num_snspds - 2])
    D = pg.union(D, by_layer=True)
    D.add_port(name=3*num_snspds, port=clk_ro_taper.ports[2])
    shiftreg_clk_ports = []
    shiftreg_heater_ports = []
    for i in range(num_snspds):
        D.add_port(name=3*i, port=snspds[i].ports[0])
        D.add_port(name=3*i+1, port=snspds[i].ports[2])
        D.add_port(name=3*i+2, port=snspds[i].ports[4])
        shiftreg_clk_ports.append(D.ports[3*i+1])
        if i < num_snspds - 1:
            shiftreg_clk_ports.append(D.ports[3*i+2])
        shiftreg_heater_ports.append(D.ports[3*i])
    # create biasing network
    if onchip_bias:
        BS = Device("bias_superconductors")
        BR = Device("bias_resistors")
        offset = 0
        clk_res_connector = rg.optimal_tee(width=(routing_w/2, routing_w), side_length=2*routing_w, layer=nbn_layer)
        max_height = max(2*routing_w*(len(shiftreg_clk_ports) + 1)//2 + 3*routing_w, 9*routing_w)
        clk_bias_res = rg.resistor_with_vias(via_layer=via_layer, res_layer=heater_layer, res_w=2.5*wire_w,
                                             res_sq=50, via_max_w=None, max_height=max_height,
                                             meander_spacing=1)
        phase1_h, phase2_h = None, None
        left_shunt_ports = []
        right_shunt_ports = []
        for p in range((len(shiftreg_clk_ports) + 1)//2):
            turn = rg.optimal_l(width=routing_w, side_length=5*routing_w + offset, layer=nbn_layer)
            b1 = BS << clk_res_connector
            r1 = BR << clk_bias_res
            t1 = BS << turn
            b1.connect(b1.ports[2], shiftreg_clk_ports[p])
            t1.connect(t1.ports[1], shiftreg_clk_ports[p])
            b1.move((t1.xmax - b1.xmin - routing_w/2, 2*routing_w if p % 2 == 1 else 0))
            r1.mirror((0,0), (0,1))
            r1.connect(r1.ports[1], b1.ports[3])
            r1.rotate(angle=90, center=r1.ports[1].midpoint)
            r1.move((-2.5*wire_w + 1, -wire_w/2))
            if p % 2 == 0:
                phase1_h = r1.ports[2].y - wire_w/2
            else:
                phase2_h = r1.ports[2].y - wire_w/2
            # add straight
            s1 = BS << pg.straight(size=(routing_w, t1.ports[2].x - BS.xmin), layer=nbn_layer)
            s1.connect(s1.ports[1], t1.ports[2])
            left_shunt_ports.append(s1.ports[2])
            left_shunt_ports.append(t1.ports[1])
            if p < (len(shiftreg_clk_ports) + 1)//2 - 1:
                b2 = BS << clk_res_connector
                r2 = BR << clk_bias_res
                t2 = BS << turn
                b2.connect(b2.ports[1], shiftreg_clk_ports[len(shiftreg_clk_ports) - 1 - p])
                t2.connect(t2.ports[2], shiftreg_clk_ports[len(shiftreg_clk_ports) - 1 - p])
                b2.move((t2.xmin - b2.xmax + routing_w/2, 2*routing_w if (len(shiftreg_clk_ports) - 1 - p) % 2 == 1 else 0))
                r2.connect(r2.ports[1], b2.ports[3])
                r2.rotate(angle=-90, center=r2.ports[1].midpoint)
                r2.move((2.5*wire_w - 1, -wire_w/2))
                # add straight
                s2 = BS << pg.straight(size=(routing_w, BS.xmax - t2.ports[1].x), layer=nbn_layer)
                s2.connect(s2.ports[1], t2.ports[1])
                right_shunt_ports.append(s2.ports[2])
                right_shunt_ports.append(t2.ports[2])
            offset += 2*routing_w
        # add shared clk_res_connector for each clock
        bias_len = BS.xsize
        bias_xmin = BS.xmin
        bias = pg.straight(size=(routing_w, bias_len), layer=nbn_layer).rotate(90)
        phase1 = BS << bias
        phase2 = BS << bias
        phase1.move((bias_xmin - phase1.xmin, phase1_h - phase1.y))
        phase2.move((bias_xmin - phase2.xmin, phase2_h - phase2.y))
        # create bias resistors for "snspds"/heaters
        heater_bias_res = rg.resistor_with_vias(via_layer=via_layer, res_layer=heater_layer, res_w=3*wire_w,
                                                res_sq=30, via_max_w=(routing_w, routing_w), max_height=None,
                                                meander_spacing=1)
        res_pad = pg.straight(size=(routing_w, routing_w), layer=nbn_layer)
        heater_ports = []
        for p in range(len(shiftreg_heater_ports)):
            h = BR << heater_bias_res
            h.connect(h.ports[1], shiftreg_heater_ports[p])
            h.move((0, -routing_w))
            r = BS << res_pad
            r.connect(r.ports[1], h.ports[2])
            r.move((0, -routing_w/2 - h.ports[2].width/2))
            heater_ports.append(r.ports[2])
        # merge superconducting network, create ports, and make outline
        BS = pg.union(BS)
        bs_ports = []
        for n, port in enumerate(left_shunt_ports):
            BS.add_port(name=n, port=port)
            if n % 2 == 0:
                bs_ports.append(port)
        offset = len(left_shunt_ports)
        BS.add_port(name=offset, port=phase1.ports[1])
        bs_ports.append(phase1.ports[1])
        offset += 1
        for n, port in enumerate(heater_ports):
            BS.add_port(name=n + offset, port=port)
            bs_ports.append(port)
        offset += len(heater_ports)
        BS.add_port(name=offset, port=phase2.ports[2])
        bs_ports.append(phase2.ports[2])
        offset += 1
        for n, port in enumerate(reversed(right_shunt_ports)):
            BS.add_port(name=n + offset, port=port)
            if n % 2 == 1:
                bs_ports.append(port)
        BS = pg.outline(BS, distance=dev_outline, open_ports=True, layer=nbn_layer)
        D << BR
        bs = D << BS
        readout_ports = [D.ports[len(D.ports)-2], D.ports[len(D.ports)-1]]
        D = pg.union(D, by_layer=True)
        for n, port in enumerate(bs_ports):
            D.add_port(name=n, port=port)
        D.add_port(name=len(bs_ports), port=readout_ports[0])
        D.add_port(name=len(bs_ports) + 1, port=readout_ports[1])
    return D

def make_device_pair(onchip_bias = True,
                     snspd_count = 3,
                     dev_outline = 0.2,
                     pad_outline = 10,
                     nbn_pad_layer = 1,
                     nbn_layer = 0,
                     via_layer = 2,
                     heater_layer = 3,
                     snspd_w = 0.3,
                     snspd_ff = 0.3,
                     snspd_sq = 10000,
                     heater_w = 3,
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
    E = Device('experiment')
    R = Device('routes')
    shiftregs = []
    port_offset = 0
    for snspd in (True, False):
        if snspd:
            pitch = snspd_w/snspd_ff
            snspd_size = (pitch*snspd_w*snspd_sq)**0.5
            ind_width = 50*wire_w if ind_sq > 500 else (30*wire_w if ind_sq > 200 else 20*wire_w)
            spacing = max(snspd_size/2 - ind_width + 5, 10)
        else:
            spacing = 30
        shiftreg = E << shiftreg_snspd_row(nbn_layer=nbn_layer, via_layer=via_layer, heater_layer=heater_layer,
                                           onchip_bias=onchip_bias if not snspd else False, snspd=snspd,
                                           snspd_w=snspd_w, snspd_ff=snspd_ff, snspd_sq=snspd_sq,
                                           heater_w=heater_w, stage_count=2*snspd_count - 1, ind_spacing=spacing,
                                           readout_channel_w=readout_channel_w, readout_gate_w=readout_gate_w,
                                           readout_source_sq=readout_source_sq, readout_drain_sq=readout_drain_sq,
                                           ind_sq=ind_sq, dev_outline=dev_outline, wire_w=wire_w,
                                           htron_constriction_ratio=htron_constriction_ratio,
                                           constriction_w=constriction_w, routing_w=routing_w, drain_sq=drain_sq,
                                           source_sq=source_sq)
        if not snspd:
            shiftreg.rotate(180)
            shiftreg.move((shiftregs[0].x - shiftreg.x, shiftregs[0].ymin - shiftreg.ymax - 10*routing_w))
        shiftregs.append(shiftreg)
        # add ports
        for pid, port in shiftreg.ports.items():
            E.add_port(name=port_offset+pid, port=port)
        port_offset += len(shiftreg.ports.items())
    E.move((-E.x, -E.y))
    exp = D << E
    new_exp_ports = []
    for i in range(len(exp.ports)):
        exp_port = exp.ports[i]
        ep_n = exp_port.normal[1] - exp_port.center
        if np.dot(ep_n, (1,0)) == 0:
            if np.dot(ep_n, (0,1)) > 0:
                # up
                straight_len = exp.ymax - exp_port.center[1]
            else:
                # down
                straight_len = exp_port.center[1] - exp.ymin 
        else:
            if np.dot(ep_n, (1,0)) > 0:
                # right
                straight_len = exp.xmax - exp_port.center[0]
            else:
                # left
                straight_len = exp_port.center[0] - exp.xmin
        if straight_len > 0:
            s = pg.straight(size=(routing_w, straight_len))
            ol = pg.outline(s, distance=dev_outline, open_ports=True, layer=nbn_layer)
            to_edge = R << ol
            to_edge.connect(to_edge.ports[1], exp_port)
            to_edge.ports[2].name = exp_port.name
            new_exp_ports.append(to_edge.ports[2])
    output_ports = [len(shiftregs[0].ports) - 2]
    output_ports.append(output_ports[0] + len(shiftregs[1].ports))
    pad_count = len(exp.ports)
    #workspace_size = (1.5*E.xsize, 1.3*E.ysize)
    #workspace_size = (1.8*E.xsize, 1.8*E.ysize)
    workspace_size = (1.6*E.xsize, 2.0*E.ysize)
    #workspace_size = (2.5*E.xsize, 2.5*E.ysize)
    pad_array = rg.pad_array(num_pads=pad_count, workspace_size=workspace_size,
                             pad_layers=tuple(nbn_pad_layer for i in range(pad_count)),
                             outline=pad_outline,
                             pad_size=(200,250),
                             pos_tone= {nbn_pad_layer: True})
    pa = D << pad_array
    routes = rg.autoroute(exp_ports=new_exp_ports, pad_ports=list(sorted(pa.ports.values(), key=lambda v: v.name)),
                          workspace_size=workspace_size, exp_bbox=exp.bbox, width=routing_w, spacing=2*routing_w,
                          pad_offset=pad_outline+3*routing_w, layer=nbn_layer)
    routes = R << pg.outline(routes, distance=dev_outline, open_ports=True, layer=nbn_layer)
    qp([exp, pad_array])
    qp(routes)
    D << R
    D = pg.union(D, by_layer=True)
    return D
    

D = Device("test")
#D << multistage_shiftreg(ind_spacing=3, stage_count=3, ind_sq=200)
#D << shiftreg_snspd_row(onchip_bias=True, snspd=False, ind_sq=500, ind_spacing=30, stage_count=7)
#D << shiftreg_snspd_row(stage_count=3, onchip_bias=False, snspd=True, ind_sq=500, snspd_w=0.3, snspd_sq=10000, snspd_ff=0.2, wire_w=1, heater_w=3, routing_w=5)
D << make_device_pair(onchip_bias=True, snspd_count=2, ind_sq=500, snspd_w=0.3, snspd_sq=20000, snspd_ff=0.2, wire_w=1, heater_w=3, routing_w=5)
#D << make_device_pair(onchip_bias=False, snspd_count=3, ind_sq=500, snspd_w=0.3, snspd_sq=10000, snspd_ff=0.2, wire_w=1, heater_w=3, routing_w=5)
qp(D)
input('press any key to exit')
exit()
#D << pg.optimal_step(0.1, 1, symmetric=True)
#D << rg.optimal_tee()
#for ind_sq in [200]:
for ind_sq in [200, 500, 1000]:
#for ind_sq in [500, 1000, 2000]:
    #D << shiftreg_snspd_row(ind_sq=ind_sq, ind_spacings=(20, 5, 20, 5, 20), wire_w=1, snspd=True, snspd_sq=15000, dev_outline=0.45)
    #D << shiftreg_snspd_row(ind_sq=ind_sq, ind_spacings=(20, 5, 20, 5, 20), wire_w=1, snspd=False)
    D << make_device_pair(ind_sq=ind_sq, ind_spacings=(20, 20, 20, 20, 20), snspd_sq=15000, dev_outline=0.45)
    D << make_device_pair(ind_sq=ind_sq, ind_spacings=(10, 5, 10, 5, 10), snspd_sq=10000, snspd_w=0.3, dev_outline=0.45)
    D << make_device_pair(ind_sq=ind_sq, ind_spacings=(10, 5, 10), snspd_sq=10000, snspd_w=0.3, dev_outline=0.45)
    #D << multistage_shiftreg()
D.distribute(direction = 'y', spacing = 10)
#D.write_gds('shiftreg_snspd.gds', unit=1e-06, precision=1e-09, auto_rename=True, max_cellname_length=1024, cellname='toplevel')
qp(D)

input('press any key to exit')
