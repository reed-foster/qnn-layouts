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
import phidl.path as pp
from phidl import quickplot as qp
from phidl import set_quickplot_options
set_quickplot_options(blocking=True, show_ports=True)

import qnngds.utilities as qu
import qnngds.geometry as qg

def shiftreg_halfstage(loop_sq = 2000,
                       loop_spacing = 0,
                       drain_sq = 200,
                       wire_w = 0.4, # much larger than constriction
                       ntron_gate_w = None,
                       constriction_w = 0.122, # t=16nm, Jc=46GA/m^2, Ic=90uA
                       nanowire_sq = 5, # don't make too many squares (fab limits)
                       switch_type = 'nw',
                       separate_shunt = 1,
                       layer = 1):
    """
    single halfstage
    
    ntron or current-summing nanowire and series/loop inductor
    parameters:
    loop_sq         - number of loop inductor squares
    loop_spacing    - spacing between loop inductor and drain bias (in um)
    drain_sq        - number of drain inductor squares 
    wire_w          - width of gate, drain, source, and inductors (in um)
    ntron_gate_w    - width of ntron choke width (ignored for 'nw' switch_type)
    constriction_w  - channel width (or width of nanowire constriction)
    nanowire_sq     - number of squares for nanowire
    switch_type     - 'ntron' or 'nw'
    separate_shunt  - 1 if a separate contact for shunting resistor is constructed, 0 otherwise
    layer           - layer
    """
    if switch_type not in ['nw', 'ntron']:
        raise ValueError(f'Invalid switch_type {switch_type} (choose: nw/ntron)')
    D = Device('sr_halfstage')
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
        taper_0 = pg.optimal_step(wire_w, constriction_w, symmetric=True)
        taper_1 = pg.optimal_step(wire_w, constriction_w, symmetric=True)
        source = qg.hyper_taper(4*wire_w, 4*wire_w, wire_w)
        t0 = D << taper_0
        t1 = D << taper_1
        s = D << source
        t0.connect(taper_0.ports[2], nw.ports[1])
        t1.connect(taper_1.ports[2], nw.ports[2])
        s.connect(source.ports[1], t1.ports[1])

        # create junction for connecting gate and drain inductor to nanowire
        junction = pg.tee(size=(10*wire_w,wire_w), stub_size=(wire_w,10*wire_w), taper_type='fillet')
        jct = D << junction
        jct.connect(junction.ports[1], t0.ports[1])
        
        # extra wire so that cascaded half-stages line up
        drain_wire = pg.straight(size=(wire_w, 2*wire_w))
        dw = D << drain_wire
        dw.connect(drain_wire.ports[1], jct.ports[2])
    else:
        if ntron_gate_w is None:
            ntron_gate_w = constriction_w
        ntron = qg.ntron_sharp(choke_w=ntron_gate_w, choke_l=wire_w, gate_w=wire_w,
                               channel_w=constriction_w, source_w=wire_w, drain_w=wire_w)
        nt = D << ntron
        src_wire = pg.straight(size=(wire_w, 5*wire_w))
        sw = D << src_wire
        sw.connect(src_wire.ports[1], nt.ports['s'])
        source = qg.hyper_taper(4*wire_w, 4*wire_w, wire_w)
        s = D << source
        s.connect(source.ports[1], sw.ports[2])
        # offset the vertical dimension of the ntron choke so that inputs and outputs
        # of cascaded half-stages line up
        drain_wire = pg.straight(size=(wire_w, 7*wire_w - 6.05*constriction_w))
        dw = D << drain_wire
        dw.connect(drain_wire.ports[1], nt.ports['d'])
    #####################################
    # create common structures
    #####################################
    # create gate
    gate = pg.straight(size=(wire_w,0 if switch_type == 'nw' else 10*wire_w))
    g = D << gate
    if switch_type == 'nw':
        g.connect(gate.ports[1], jct.ports[3])
    else:
        g.connect(gate.ports[1], nt.ports['g'])
    # create drain inductor
    if drain_sq < 40:
        # just make a straight that is drain_sq*wire_w tall
        drain_ind = pg.straight(size=(wire_w, drain_sq*wire_w))
    else:
        drain_ind = pg.snspd(wire_width=wire_w, wire_pitch=wire_w*2,
                             size=(20*wire_w, None), num_squares=drain_sq)
    ld = D << drain_ind
    if switch_type == 'nw':
        ld.connect(drain_ind.ports[1], dw.ports[2])
    else:
        ld.connect(drain_ind.ports[1], dw.ports[2])

    # create junction for connecting bias and loop inductor to summing structure
    bias_junction = pg.tee(size=(10*wire_w,wire_w), stub_size=(wire_w,10*wire_w), taper_type='fillet')
    bjct = D << bias_junction
    bjct.connect(bias_junction.ports[2], ld.ports[2])
    if separate_shunt:
        sjct = D.add_ref(bias_junction)
        sjct.connect(sjct.ports[1], ld.ports[2])
        shunt_taper = qg.hyper_taper(4*wire_w, 4*wire_w, wire_w)
        sh_taper = D << shunt_taper
        sh_taper.connect(shunt_taper.ports[1], sjct.ports[3])

    # create drain bias
    drain = qg.hyper_taper(4*wire_w, 4*wire_w, wire_w)
    d = D << drain
    d.connect(drain.ports[1], bjct.ports[1])

    # create loop inductor
    loop_ind = pg.snspd(wire_width=wire_w, wire_pitch=wire_w*2,
                        size=(None, 32*wire_w), num_squares=loop_sq)
    ll = D << loop_ind
    if loop_spacing > 10*wire_w:
        ind_connector_0 = pg.straight(size=(wire_w, loop_spacing - 10*wire_w))
        ic0 = D << ind_connector_0
        ic0.connect(ind_connector_0.ports[1], bjct.ports[3])
        ll.connect(loop_ind.ports[1], ic0.ports[2])
    else:
        ll.connect(loop_ind.ports[1], bjct.ports[3])
    ind_connector_1 = pg.straight(size=(wire_w, loop_spacing))
    ic1 = D << ind_connector_1
    ic1.connect(ind_connector_1.ports[1], ll.ports[2])
    
    # merge shapes and create new ports
    D = pg.union(D)
    D.flatten(single_layer=layer)
    D.add_port(name='gnd', port=s.ports[2])
    D.add_port(name='input', port=g.ports[2])
    D.add_port(name='drain', port=d.ports[2])
    if separate_shunt:
        D.add_port(name='shunt', port=sh_taper.ports[2])
    D.add_port(name='output', port=ic1.ports[2])
    D.name = 'sr_halfstage'
    D.info = locals()
    return D

def shiftreg_readout(num_halfstages = 4,
                     loop_sq = 2000,
                     loop_spacing = 5,
                     drain_sq = 200,
                     wire_w = 0.4, # much larger than constriction
                     constriction_w = 0.122, # t=16nm, Jc=46GA/m^2, Ic=90uA
                     nanowire_sq = 5, # don't make too many squares (fab limits)
                     switch_type = 'nw',
                     separate_shunt = 1,
                     final_loop_sq = 1800,
                     final_loop_spacing = 0,
                     term_gate_w = 0.015,
                     term_channel_w = 0.13,
                     term_drain_sq = 200,
                     layer = 1):
    """
    multiple halfstages + readout circuit
   
    ntron for destroying circulating current of final stage of shift register
    of first ntron.
    parameters:
    num_halfstages      - number of halfstages to construct
    loop_sq             - number of loop inductor squares
    final_loop_sq       - number of loop inductor squares for final loop
    loop_spacing        - spacing between loop inductor and drain bias (in um)
    final_loop_spacing  - spacing between loop inductor and drain bias (in um)
    drain_sq            - number of drain inductor squares 
    wire_w              - width of gate, drain, source, and inductors (in um)
    constriction_w      - channel width (or width of nanowire constriction)
    nanowire_sq         - number of squares for nanowire
    switch_type         - 'ntron' or 'nw'
    separate_shunt      - 1 if a separate contact for shunting resistor is constructed, 0 otherwise
    term_gate_w         - width of termination ntron gate (in um)
    term_channel_w      - width of termination ntron channel (in um)
    layer               - layer
    """
    D = Device('sr_readout')

    # create halfstages
    halfstages = []
    hs = shiftreg_halfstage(loop_sq=loop_sq, loop_spacing=loop_spacing,
                            drain_sq=drain_sq, wire_w=wire_w,
                            constriction_w=constriction_w, nanowire_sq=nanowire_sq,
                            switch_type=switch_type, separate_shunt=separate_shunt,
                            layer=layer)
    # separate number of squares and spacing for final loop
    final_hs = shiftreg_halfstage(loop_sq=final_loop_sq, loop_spacing=final_loop_spacing,
                                  drain_sq=drain_sq, wire_w=wire_w,
                                  constriction_w=constriction_w, nanowire_sq=nanowire_sq,
                                  switch_type=switch_type, separate_shunt=separate_shunt,
                                  layer=layer)
    # generate stages
    for i in range(num_halfstages):
        if i == num_halfstages - 1:
            hs_ref = D.add_ref(final_hs)
        else:
            hs_ref = D.add_ref(hs)
        halfstages.append(hs_ref)
        # connect adjacent stages
        if i > 0:
            hs_ref.connect(hs_ref.ports['input'], halfstages[i-1].ports['output'])

    # add termination
    # a tee to connect output of last halfstage to series resistor
    # and drain of termination ntron
    termination_tee = pg.tee(size=(10*wire_w,wire_w), stub_size=(wire_w,10*wire_w),
                             taper_type='fillet')
    term_tee = D << termination_tee
    term_tee.connect(termination_tee.ports[3], halfstages[-1].ports['output'])
    # ntron to destroy circulating current in final stage
    t_ntron = qg.ntron_sharp(choke_w=term_gate_w, choke_l=term_gate_w, gate_w=term_gate_w,
                             channel_w=term_channel_w, source_w=wire_w, drain_w=wire_w)
    tnt = D << t_ntron
    tnt.connect(t_ntron.ports['s'], term_tee.ports[1])
    # connect drain of termination ntron to series resistor (off-chip)
    term_to_res = qg.hyper_taper(4*wire_w, 4*wire_w, wire_w)
    t2r = D << term_to_res
    t2r.connect(term_to_res.ports[1], term_tee.ports[2])
    # add gate connection to termination ntron
    term_gate_taper = qg.hyper_taper(4*wire_w, 4*wire_w, term_gate_w)
    term_gt = D << term_gate_taper
    term_gt.connect(term_gate_taper.ports[1], tnt.ports['g'])
    # add connection to groundplane
    term_src_taper = qg.hyper_taper(4*wire_w, 4*wire_w, wire_w)
    term_st = D << term_src_taper
    term_st.connect(term_src_taper.ports[1], tnt.ports['d'])

    D = pg.union(D)
    D.flatten(single_layer=layer)
    for i in range(num_halfstages):
        D.add_port(name=f'drain_{i}', port=halfstages[i].ports['drain'])
        D.add_port(name=f'gnd_{i}', port=halfstages[i].ports['gnd'])
        if separate_shunt:
            D.add_port(name=f'shunt_{i}', port=halfstages[i].ports['shunt'])
    D.add_port(name='input', port=halfstages[0].ports['input'])
    D.add_port(name='term_drain', port=t2r.ports[2])
    D.add_port(name='term_gnd', port=term_st.ports[2])
    D.add_port(name='term_gate', port=term_gt.ports[2])
    D.name = 'sr_readout'
    D.info = locals()
    return D

def shiftreg(input_gate_w = 0.02,
             input_channel_w = 0.32,
             input_stage = 1,
             num_halfstages = 4,
             loop_sq = 2000,
             loop_spacing = 5,
             drain_sq = 200,
             wire_w = 0.4, # much larger than constriction
             constriction_w = 0.122, # t=16nm, Jc=46GA/m^2, Ic=90uA
             nanowire_sq = 5, # don't make too many squares (fab limits)
             switch_type = 'nw',
             separate_shunt = 1,
             final_loop_sq = 1800,
             final_loop_spacing = 0,
             term_gate_w = 0.015,
             term_channel_w = 0.13,
             term_drain_sq = 200,
             amp_gate_w = 0.012,
             amp_channel_w = 0.13,
             amp_gate_sq = 2000,
             dev_outline = 0.2,
             pad_outline = 10,
             dev_layer = 1,
             pad_layer = 2):
    """
    full shift register with input/output stages and pads

    parameters:
    input_gate_w        - width of input gate (um)
    input_channel_w     - width of input channel (um)
    input_stage         - 1 if input stage is included, 0 if not
    num_halfstages      - number of halfstages to construct
    loop_sq             - number of loop inductor squares
    final_loop_sq       - number of loop inductor squares for final loop
    loop_spacing        - spacing between loop inductor and drain bias (in um)
    final_loop_spacing  - spacing between loop inductor and drain bias (in um)
    drain_sq            - number of drain inductor squares 
    wire_w              - width of gate, drain, source, and inductors (in um)
    constriction_w      - channel width (or width of nanowire constriction)
    nanowire_sq         - number of squares for nanowire
    switch_type         - 'ntron' or 'nw'
    separate_shunt      - 1 if a separate contact for shunting resistor is constructed, 0 if not
    term_gate_w         - width of termination ntron gate (in um)
    term_channel_w      - width of termination ntron channel (in um)
    amp_gate_w          - width of amplifier ntron gate (in um)
    amp_channel_w       - width of amplifier ntron channel (in um)
    amp_gate_sq         - number of squares on amplifier ntron gate
    dev_outline         - outline thickness (in um) for devices
    pad_outline         - outline thickness (in um) for pads
    dev_layer           - device layer
    pad_layer           - pad layer
    """
    D = Device('shiftreg template')

    # create stages + termination ntron
    stages = shiftreg_readout(num_halfstages=num_halfstages, loop_sq=loop_sq,
                              loop_spacing=loop_spacing, drain_sq=drain_sq,
                              wire_w=wire_w, constriction_w=constriction_w,
                              nanowire_sq=nanowire_sq, switch_type=switch_type,
                              separate_shunt=separate_shunt, final_loop_sq=final_loop_sq,
                              final_loop_spacing=final_loop_spacing,
                              term_gate_w=term_gate_w, term_channel_w=term_channel_w,
                              term_drain_sq=term_drain_sq, layer=dev_layer)
    s = D << stages
    
    if input_stage:
        # add input stage
        input_stage = shiftreg_halfstage(loop_sq=loop_sq, loop_spacing=loop_spacing,
                                         drain_sq=10, wire_w=wire_w, ntron_gate_w=input_gate_w,
                                         constriction_w=input_channel_w, nanowire_sq=nanowire_sq,
                                         switch_type='ntron', separate_shunt=separate_shunt,
                                         layer=dev_layer)
        i_stage = D << input_stage
        i_stage.connect(input_stage.ports['output'], stages.ports['input'])

    # get number of pads required
    drain_shunt_pads = num_halfstages * (2 if separate_shunt else 1)
    input_pads = (3 if separate_shunt else 2) if input_stage else 1
    # no shunt for readout, but separate_shunt chooses if readout line has separate terminal from bias
    readout_pads = 5 if separate_shunt else 4
    pad_count = 2*(drain_shunt_pads + input_pads + readout_pads)
    workspace_sidelength = pad_count/4*80
    pad_array = qg.pad_array(num=pad_count, size1=(workspace_sidelength, workspace_sidelength),
                             outline=pad_outline)
    
    # add amplifier ntron
    a_ntron = qg.ntron_sharp(choke_w=amp_gate_w, choke_l=amp_gate_w,
                             gate_w=wire_w, channel_w=amp_channel_w,
                             source_w=wire_w, drain_w=wire_w).mirror()
    ant = D << a_ntron.move((D.xmax+50, 0))
    amp_gate_taper = qg.hyper_taper(4*wire_w, 4*wire_w, amp_gate_w)
    amp_drain_taper = qg.hyper_taper(4*wire_w, 4*wire_w, wire_w)
    amp_source_taper = qg.hyper_taper(4*wire_w, 4*wire_w, wire_w)
    amp_gt = D << amp_gate_taper
    amp_dt = D << amp_drain_taper
    amp_st = D << amp_source_taper
    amp_gt.connect(amp_gate_taper.ports[1], ant.ports['g'])
    amp_st.connect(amp_source_taper.ports[1], ant.ports['s'])
    if separate_shunt:
        amp_tee = pg.tee(size=(20*wire_w,wire_w), stub_size=(wire_w,10*wire_w), taper_type='fillet')
        atee = D << amp_tee
        atee.connect(amp_tee.ports[2], ant.ports['d'])
        amp_dt.connect(amp_drain_taper.ports[1], atee.ports[1])
    else:
        amp_dt.connect(amp_drain_taper.ports[1], ant.ports['d'])

    # add input/output tapers
    input_taper = qg.hyper_taper(4*wire_w, 4*wire_w, wire_w)
    i_taper = D << input_taper
    if input_stage:
        i_taper.connect(input_taper.ports[1], i_stage.ports['input'])
    else:
        i_taper.connect(input_taper.ports[1], stages.ports['input'])
    output_taper = qg.hyper_taper(4*wire_w, 4*wire_w, wire_w)
    o_taper = D << output_taper
    if separate_shunt:
        o_taper.connect(output_taper.ports[1], atee.ports[3])
    else:
        o_taper.connect(output_taper.ports[1], ant.ports['d'])
    # merge everything so far
    D = pg.union(D)
    D.flatten(single_layer=dev_layer)

    D.add_port(name=0, port=i_taper.ports[2])
    if input_stage:
        if separate_shunt:
            D.add_port(name=1, port=i_stage.ports['shunt'])
            D.add_port(name=2, port=i_stage.ports['drain'])
        else:
            D.add_port(name=1, port=i_stage.ports['drain'])
    # give gnd ports separate name so we can ignore them later
    port_id_offset = input_pads
    for i in range(num_halfstages):
        if separate_shunt:
            drain_id = 2*i + 1 + port_id_offset
            shunt_id = 2*i + port_id_offset
            D.add_port(name=shunt_id, port=stages.ports[f'shunt_{i}'])
        else:
            drain_id = i + port_id_offset
        D.add_port(name=drain_id, port=stages.ports[f'drain_{i}'])
    port_id_offset += 2*num_halfstages if separate_shunt else num_halfstages
    D.add_port(name=port_id_offset, port=stages.ports['term_drain'])
    D.add_port(name=port_id_offset+1, port=stages.ports['term_gate'])
    port_id_offset += 2
    if separate_shunt:
        D.add_port(name=port_id_offset, port=amp_dt.ports[2])
        port_id_offset += 1
    D.add_port(name=port_id_offset, port=o_taper.ports[2])
    D.add_port(name=port_id_offset + 1, port=amp_gt.ports[2])
    # now do ground ports so that they can be separated later and ignored
    # when auto connecting device ports to the pad array
    if input_stage:
        D.add_port(name=port_id_offset + 2, port=i_stage.ports['gnd'])
    port_id_offset += 3 if input_stage else 2
    for i in range(num_halfstages):
        D.add_port(name=port_id_offset+i, port=stages.ports[f'gnd_{i}'])
    port_id_offset += num_halfstages
    D.add_port(name=port_id_offset, port=stages.ports['term_gnd'])
    D.add_port(name=port_id_offset + 1, port=amp_st.ports[2])
    D = pg.outline(D, distance=dev_outline, open_ports=True)
    # create new Device which will hold ntrons/nanowires and pads
    D2 = Device('shiftreg_layer')
    # add two references and move them to their respective halves of the pad array
    sr0 = D2.add_ref(D).translate(-D.x,-D.y).translate(0, workspace_sidelength/8)
    sr1 = D2.add_ref(D).translate(-D.x,-D.y).rotate(angle=180).translate(0, -workspace_sidelength/8)
    num_not_gnd = len(sr0.ports) - num_halfstages - (3 if input_stage else 2)
    # create ports on references
    for port_id in sr0.ports.keys():
        if port_id < num_not_gnd:
            # skip labeling ground ports
            new_id = port_id
            D2.add_port(name=new_id, port=sr0.ports[port_id])
    port_id_offset = num_not_gnd
    for port_id in sr1.ports.keys():
        if port_id < num_not_gnd:
            # skip labeling ground ports
            new_id = port_id + port_id_offset
            D2.add_port(name=new_id, port=sr1.ports[port_id])
    # route nw connections to pads
    we_count = pad_count // 4
    ns_count = (pad_count + 3) // 4
    # renumber pads
    remapped_pads = []
    # first get upper half of west edge
    for i in range(we_count//2):
        remapped_pads.append(pad_count-we_count//2+i)
    # north edge
    for i in range(ns_count):
        remapped_pads.append(i)
    # east edge
    for i in range(we_count):
        remapped_pads.append(pad_count-we_count-1-i)
    # south edge
    for i in range(ns_count):
        remapped_pads.append(2*ns_count-1-i)
    # lower half of west edge
    for i in range((we_count+1)//2):
        remapped_pads.append(pad_count-we_count+i)
    pads = D2 << pad_array
    port_id_offset = num_not_gnd * 2
    for i in range(pad_count):
        D2.add_port(name=i + port_id_offset, port=pads.ports[remapped_pads[i]])
    # autoconnect ports
    for i in range(pad_count):
        R = pr.route_smooth(D2.ports[i], D2.ports[i + port_id_offset], width = 4*wire_w, radius = 5)
        D2 << R
    D2.name = 'full_shiftreg'
    D2.info = locals()
    return D2

D = Device()

#D << shiftreg_halfstage(switch_type='ntron', drain_sq=200)
#D << shiftreg_halfstage(switch_type='nw', drain_sq=200)
#D << shiftreg_halfstage(switch_type='ntron').move((0,2.6+0.125))
#D << qg.pad_array(num=18, size1=(400, 400), outline=10)
#D << pg.outline(shiftreg_readout(2, switch_type = 'nw'), distance=0.2, open_ports=True)
D << shiftreg(num_halfstages=1, switch_type='ntron', separate_shunt = 0, input_stage = 0)
#D.align(alignment = 'ymax')
#D.distribute(direction = 'x', spacing = 30)
qp(D)

#D << qg.ntron_three_port()

