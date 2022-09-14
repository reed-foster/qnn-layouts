# -*- coding: utf-8 -*-
"""
Created on Wed Mar 02 12:08:33 2022

@author: reedf

"""
import traceback

import numpy as np
from phidl import Device
from phidl import Path
from phidl import Port
import phidl.geometry as pg
import phidl.routing as pr
import phidl.path as pp

from phidl import quickplot as qp
from phidl import set_quickplot_options
#set_quickplot_options(blocking=True, show_ports=True)
set_quickplot_options(blocking=False, show_ports=True, new_window=True)

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
    swap = False
    if width[0] >= width[1]:
        # use >= so the default case uses the correct labeling scheme
        width = (width[1], width[0])
        p1, p2 = p2, p1
        swap = True

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
    if not swap:
        D.mirror((0,0), (0,1))
        D.rotate(270)
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

def resistor_with_vias(via_layer = 1,
                       res_layer = 2,
                       res_w = 3,
                       res_sq = 1,
                       via_max_w = (1, None),
                       max_height = None,
                       meander_spacing = 1):
    """
    creates a resistor with via connections

    via_layer       - gds layer for vias
    res_layer       - gds layer for resistor
    res_w           - resistor width in microns
    res_sq          - number of squares in resistor (between vias)
    via_max_w       - tuple, maximum via width for each port (if none, then use res_w - 1)
    max_height      - max height of resistor in microns (create meanders if max_height is not None and res_w*res_sq > max_height)
    meander_spacing - spacing between meanders in number of squares
    """

    D = Device('resistor_with_vias')

    # check arguments
    if via_max_w is None:
        via_max_w = (None, None)
    if len(via_max_w) != 2:
        raise ValueError("invalid number of via max widths, please use None or (width1/None, width2/None)")

    # determine meander count based on max_height
    n_meander = 0
    hp_n = lambda n: res_w/2*(res_sq - 2*(1 + meander_spacing)*n - (2*n - 1)*height/res_w)
    if max_height is None or res_w*res_sq < max_height:
        height = res_w*res_sq
    else:
        height = max_height
        # num squares = (2*
        while hp_n(n_meander) > height:
            n_meander += 1
        if hp_n(n_meander) < 0:
            n_meander -= 1


    vias = []
    for n, v_wmax in enumerate(via_max_w):
        via_w = min(v_wmax, res_w - 1) if v_wmax is not None else res_w - 1
        v = D << pg.rectangle(size=(via_w, via_w), layer=via_layer)
        dx = -v.xmax - height/2 if n == 0 else v.xmin + height/2
        v.move((dx, -v.y))
        vias.append(v)
    vias[1].move((0, 2*(1 + meander_spacing)*n_meander*res_w))
    res_long = pg.rectangle(size=(height + 2*res_w, res_w), layer=res_layer)
    res_short = pg.rectangle(size=(hp_n(n_meander) + 2*res_w, res_w), layer=res_layer)
    conn = pg.rectangle(size=(res_w, (2 + meander_spacing)*res_w), layer=res_layer)
    # handle case where no meander is used
    if n_meander == 0:
        r = D << res_long
        r.move((-r.x, -r.y))
    # create max length meanders
    for i in range(2*n_meander - 1):
        r = D << res_long
        c = D << conn
        dy = 2*(1 + meander_spacing)*n_meander*res_w - (1 + meander_spacing)*res_w*i
        r.move((-r.x, -r.y + dy))
        if i % 2 == 0:
            c.move((vias[0].xmax - c.xmax, -c.ymax + res_w/2 + dy))
        else:
            c.move((vias[1].xmin - c.xmin, -c.ymax + res_w/2 + dy))
    # create short meanders
    if n_meander > 0:
        for i in range(2):
            r = D << res_short
            r.move((vias[0].xmax - r.xmin - res_w, -r.y + (1 + meander_spacing)*res_w*i))
            if i == 0:
                c = D << conn
                c.move((vias[0].xmax - c.xmin + hp_n(n_meander), -c.ymin - res_w/2))
    #D.move((-D.x, -D.y))
    D = pg.union(D, by_layer=True)
    D.add_port(name=1, midpoint=(vias[0].xmin, vias[0].y), width=vias[0].ysize, orientation=180)
    D.add_port(name=2, midpoint=(vias[1].xmax, vias[1].y), width=vias[1].ysize, orientation=0)
    return D

def pad_array(num_pads = 8,
              workspace_size = (100, 200),
              pad_size = (200, 250),
              pad_layers = (2,2,1,2,2,2,2,2),
              outline = 10,
              pos_tone = {1: False, 2: True}):
    """
    rewrite of qnngds pad_array
    
    renumbers ports so they follow standard ordering (CW increasing from top left corner)
    option to do negative tone for specific pads on specific layers
    also optimizes pad count on each side to better fit workspaces with different aspect ratios
    num_pads        - total number of pads (will be evenly divided between the 4 sides
    workspace_size  - tuple, width and height (in microns) of workspace area
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
        raise ValueError(ex_str)

    min_pad_per_side, excess = divmod(num_pads, 4)
    top_bottom_pads = int(num_pads * workspace_size[0]/(workspace_size[0] + workspace_size[1]))
    left_right_pads = int(num_pads * workspace_size[1]/(workspace_size[0] + workspace_size[1]))
    pads_per_side = np.zeros(4, dtype=np.int32)
    pads_per_side[0] += left_right_pads//2 # west side
    pads_per_side[1] += left_right_pads//2 # east side
    pads_per_side[2] += top_bottom_pads//2 # south side
    pads_per_side[3] += top_bottom_pads//2 # north side
    unassigned = num_pads - 2*(top_bottom_pads//2 + left_right_pads//2)
    for i in range(unassigned):
        # prefer to put extra pads on west/east side
        pads_per_side[i] += 1
    conn_dict = {'W': pads_per_side[0],
                 'E': pads_per_side[1],
                 'S': pads_per_side[2],
                 'N': pads_per_side[3]}

    inner_compass = pg.compass_multi(size=workspace_size, ports=conn_dict, layer=1)
    outer_compass_w = max(workspace_size[0], np.max(pads_per_side[:2])*(pad_size[0] + outline*5))
    outer_compass_h = max(workspace_size[1], np.max(pads_per_side[2:])*(pad_size[0] + outline*5))
    outer_compass = pg.compass_multi(size=(outer_compass_h, outer_compass_w), ports=conn_dict, layer=1)
   
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

def autoroute(exp_ports, pad_ports, workspace_size, exp_bbox, width, spacing, pad_offset, layer):
    """
    automatically routes an experiment to a set of pads
   
    Two step process. First step partially routes any experiment ports which are orthogonal to their
    designated pad array port so that they are facing each other.  Then, sorts pairs of ports based
    on the minimum horizontal (or vertical distance) between ports facing each other. Finally, 
    routes the sorted pairs.
    exp_ports       - list of ports in experiment geometry
    pad_ports       - list of ports to connect to pad array
    workspace_size  - side length of workspace area
    exp_bbox        - bounding box of experiment
    width           - width of traces (in microns)
    spacing         - min spacing between traces (in microns)
    pad_offset      - extra spacing from pads
    layer           - gds layer
    """

    D = Device('autorouted_traces')

    if len(exp_ports) != len(pad_ports):
        raise ValueError("invalid port lists for autorouter, lengths must match")
    num_ports = len(exp_ports)

    # find the pad with the minimum distance to experiment port 0
    # we'll connect these two and connect pairs of ports sequentially around the pad array
    # so there is no overlap
    min_dist, min_pad = max(workspace_size), -1
    for i in range(num_ports):
        norm = np.linalg.norm(exp_ports[0].center - pad_ports[i].center)
        if norm < min_dist:
            min_dist, min_pad = norm, i

    # helper function for sorting pairs of ports based on their distance
    # used by both routing steps
    def pair_key(port_pair):
        ep_n = port_pair[0].normal[1] - port_pair[0].center
        pp_n = port_pair[1].normal[1] - port_pair[1].center
        if abs(np.dot(ep_n, (1,0))) < 1e-9:
            if abs(np.dot(pp_n, ep_n)) < 1e-9:
                # ports are orthogonal, and exp_port is facing up/down so sort by exp_port x
                # make negative so this gets routed first
                # we'll route these out to the left/right edge depending on the pad_port location
                return -abs(port_pair[0].x)
            else:
                # both ports are facing up/down so sort by x distance
                return abs(port_pair[0].x - port_pair[1].x)
        else:
            if abs(np.dot(pp_n, ep_n)) < 1e-9:
                # ports are orthogonal
                return -abs(port_pair[0].y)
            else:
                # source (experiment) port is facing left/right so sort by y distance
                return abs(port_pair[0].y - port_pair[1].y)
    
    # group the pairs or ports based on whether or not they are orthogonal and which direction the
    # experiment port is facing
    pairs = [(exp_ports[i], pad_ports[(min_pad + i) % num_ports]) for i in range(num_ports)]
    # split pairs into four groups based on face of experiment (N, S, E, W)
    grouped_pairs = [[], [], [], []]
    orthogonal_pairs = [[], [], [], []]
    paths = [[], [], [], []]
    for port_pair in pairs:
        ep_n = port_pair[0].normal[1] - port_pair[0].center
        pp_n = port_pair[1].normal[1] - port_pair[1].center
        if abs(np.dot(ep_n, (1,0))) < 1e-9:
            q = 0 if np.dot(ep_n, (0,1)) > 0 else 1
        else:
            q = 2 if np.dot(ep_n, (1,0)) > 0 else 3
        if abs(np.dot(pp_n, ep_n)) < 1e-9:
            orthogonal_pairs[q].append(port_pair)
        else:
            grouped_pairs[q].append(port_pair)
            paths[q].append(None)
  
    # first create partial paths for orthogonal pairs and create new port at the end of partial path
    for q, quadrant in enumerate(orthogonal_pairs):
        # keep track of height on both halves of experiment face
        # halves are based on x/y coordinate of pad_p
        # since orthogonal ports are sorted based on how close they are to the edge of the bbox
        # processing them in sorted order and incrementing height appropriately will prevent collisions
        height = [0, 0]
        for port_pair in sorted(quadrant, key = pair_key):
            exp_p = port_pair[0]
            pad_p = port_pair[1]
            if q < 2:
                # select height index based on x coordinate of pad_p
                h_idx = 0 if pad_p.x < 0 else 1
                height[h_idx] += spacing if q == 0 else -spacing
                start = (exp_p.x, exp_p.y + height[h_idx])
                end = (exp_bbox[0][0] if pad_p.x < 0 else exp_bbox[1][0], exp_p.y + height[h_idx])
                new_q = 3 if pad_p.x < 0 else 2
            else:
                # select height index based on y coordinate of pad_p
                h_idx = 0 if pad_p.y < 0 else 1
                height[h_idx] += spacing if q == 2 else -spacing
                start = (exp_p.x + height[h_idx], exp_p.y)
                end = (exp_p.x + height[h_idx], exp_bbox[0][1] if pad_p.y < 0 else exp_bbox[1][1])
                new_q = 1 if pad_p.y < 0 else 0
            path = Path((exp_p.center, start, end))
            new_port = Port(name=exp_p.name, midpoint=end, width=exp_p.width,
                        orientation=(pad_p.orientation + 180) % 360)
            grouped_pairs[new_q].append((new_port, pad_p, exp_p))
            paths[new_q].append(path)
    
    # now all ports face each other and the automatic routing will be straightforward
    for q, quadrant in enumerate(grouped_pairs):
        # keep track of height on both halves of experiment face
        # halves are based on x/y distance of pad_p and exp_p
        if q % 2 == 0:
            height = [pad_offset, pad_offset]
        else:
            height = [-pad_offset, -pad_offset]
        pad_p_prev_loc = [-1e9, 1e9] # keep track of previous pad x/y coordinate
        exp_p_prev_loc = [-1e9, 1e9]
        for p, port_pair in sorted(enumerate(quadrant), key = lambda a: pair_key(a[1])):
            exp_p = port_pair[0]
            pad_p = port_pair[1]
            if pair_key(port_pair) < pad_p.width/3:
                # pair key gets the x distance between vertically aligned ports and
                # y distance for horizontally aligned ports
                if q < 2:
                    # if exp_port is vertically aligned
                    new_path = Path((exp_p.center, (exp_p.x, pad_p.y)))
                    new_port = Port(name=pad_p.name, midpoint=(exp_p.x, pad_p.y),
                                    width=pad_p.width, orientation=(exp_p.orientation + 180) % 360)
                    pad_p_prev_loc[h_idx] = pad_p.x
                else:
                    new_path = Path((exp_p.center, (pad_p.x, exp_p.y)))
                    new_port = Port(name=pad_p.name, midpoint=(pad_p.x, exp_p.y),
                                    width=pad_p.width, orientation=(exp_p.orientation + 180) % 360)
                    pad_p_prev_loc[h_idx] = pad_p.y
            else:
                if q < 2:
                    # select height index based on difference in x coordinate (i.e. do we need to go left or right?)
                    # zero: left, one: right
                    h_idx = 0 if exp_p.x > pad_p.x else 1
                    # reset height if we're clear of adjacent routes
                    if h_idx == 0:
                        # moving leftwards
                        for pair_p in sorted(quadrant, key = lambda a: a[1].x):
                            if pair_p[1].x > pad_p.x:
                                break
                        if pair_p[1].x - pair_p[1].width/3 > exp_p.x:
                            height[h_idx] = pad_offset if q == 0 else -pad_offset
                    else:
                        # moving rightwards
                        for pair_p in sorted(quadrant, key = lambda a: -a[1].x):
                            if pair_p[1].x < pad_p.x:
                                break
                        if pair_p[1].x + pair_p[1].width/3 < exp_p.x:
                            height[h_idx] = pad_offset if q == 0 else -pad_offset
                    start = (exp_p.x, pad_p.y - height[h_idx])
                    # check if we should keep going (i.e. is |pad_p.x - exp_p.x| too small to fit a bend)
                    if h_idx == 0:
                        end_x = max(min(pad_p.x, exp_p.x - 5*width), pad_p.x - pad_p.width/3)
                    else:
                        end_x = min(max(pad_p.x, exp_p.x + 5*width), pad_p.x + pad_p.width/3)
                    mid = (end_x, pad_p.y - height[h_idx])
                    end = (end_x, pad_p.y)
                    pad_p_prev_loc[h_idx] = pad_p.x
                    exp_p_prev_loc[h_idx] = exp_p.x
                    height[h_idx] += spacing if q == 0 else -spacing
                else:
                    # select height index based on difference in y coordinate (i.e. do we need to go up or down?)
                    # zero: down, one: up
                    h_idx = 0 if exp_p.y > pad_p.y else 1
                    # reset height if we're clear of adjacent routes
                    if h_idx == 0:
                        # moving downwards
                        for pair_p in sorted(quadrant, key = lambda a: a[1].y):
                            if pair_p[1].y > pad_p.y:
                                break
                        if pair_p[1].y - pair_p[1].width/3 > exp_p.y:
                            height[h_idx] = pad_offset if q == 2 else -pad_offset
                    else:
                        # moving upwards
                        for pair_p in sorted(quadrant, key = lambda a: -a[1].y):
                            if pair_p[1].y < pad_p.y:
                                break
                        if pair_p[1].y + pair_p[1].width/3 < exp_p.y:
                            height[h_idx] = pad_offset if q == 2 else -pad_offset
                    start = (pad_p.x - height[h_idx], exp_p.y)
                    # check if we should keep going (i.e. is |pad_p.y - exp_p.y| too small to fit a bend)
                    if h_idx == 0:
                        end_y = max(min(pad_p.y, exp_p.y - 5*width), pad_p.y - pad_p.width/3)
                    else:
                        end_y = min(max(pad_p.y, exp_p.y + 5*width), pad_p.y + pad_p.width/3)
                    mid = (pad_p.x - height[h_idx], end_y)
                    end = (pad_p.x, end_y)
                    pad_p_prev_loc[h_idx] = pad_p.y
                    exp_p_prev_loc[h_idx] = exp_p.y
                    height[h_idx] += spacing if q == 2 else -spacing
                new_path = Path((exp_p.center, start, mid, end))
                new_port = Port(name=pad_p.name, midpoint=end,
                                width=pad_p.width, orientation=(exp_p.orientation + 180) % 360)
            if paths[q][p] is not None:
                paths[q][p].append(new_path)
            else:
                paths[q][p] = new_path
            grouped_pairs[q][p] = (exp_p, new_port)

    # perform routing along path and add final ports to connect to pad array
    for q, quadrant in enumerate(grouped_pairs):
        for p, port_pair in enumerate(quadrant):
            try:
                route = D << pr.route_smooth(port1=port_pair[0], port2=port_pair[1], radius=1.5*width, width=width,
                                             path_type='manual', manual_path=paths[q][p], layer=layer)
            except ValueError as e:
                traceback.print_exc()
                print('An error occurred with phidl.routing.route_smooth(), try increasing the size of the workspace')
                print(paths[q][p].points)
            # add taper
            pad_p = pad_ports[port_pair[1].name]
            final_w = port_pair[1].width + 2*(abs(route.ports[2].x - pad_p.x) + abs(route.ports[2].y - pad_p.y))
            ht = Device("chopped_hyper_taper")
            taper = ht << qg.hyper_taper(length=2*width, wide_section=final_w + width, narrow_section=width, layer=layer)
            cut = ht << pg.straight(size=(pad_p.width + width, 2*width))
            conn = ht << pg.connector(width=pad_p.width)
            taper.connect(taper.ports[1], route.ports[2])
            conn.connect(conn.ports[1], pad_p)
            cut.connect(cut.ports[1], conn.ports[1])
            D << pg.boolean(A=taper, B=cut, operation='and', precision=1e-6, layer=layer)
    D = pg.union(D)
    # create ports in final device so pg.outline doesn't block them
    for n, port in enumerate(exp_ports):
        D.add_port(name=n, midpoint=port.midpoint, width=port.width,
                   orientation=(port.orientation + 180) % 360)
    for n, port in enumerate(pad_ports):
        o = port.orientation
        dx = -2*width if o == 0 else 2*width if o == 180 else 0
        dy = -2*width if o == 90 else 2*width if o == 270 else 0
        D.add_port(name=num_ports + n, midpoint=(port.x + dx, port.y + dy), width=port.width + width,
                   orientation=(port.orientation + 180) % 360)
    return D

if __name__ == "__main__":
    # simple unit test
    D = Device("test")
    #D << qg.pad_array(pad_iso=True, de_etch=True)
    #D << qg.pad_array(num=8, outline=10, layer=2)
    #D << pad_array(num_pads=22, workspace_size=(500, 800), pad_layers=tuple(1 for i in range(22)), outline=10, pos_tone={1:True})
    D << optimal_l(width=(1,1))
    D << optimal_l(width=(1,3))
    D << optimal_l(width=(5,1))
    #D << optimal_tee(width=(1,1))
    #D << optimal_tee(width=(1,5))
    #D << pg.optimal_hairpin(width=1, pitch=1.2, length=5, turn_ratio=2, num_pts=100)
    #D << resistor_with_vias(via_layer=1, res_layer=2, res_w=5, res_sq=50, via_max_w=None, max_height=50)
    #D << resistor_with_vias(via_layer=1, res_layer=2, res_w=5, res_sq=50, via_max_w=None, max_height=80)
    #D << pg.straight(size=(4,2))
    #D << pg.straight(size=(3,9))
    D.distribute(direction = 'y', spacing = 10)
    qp(D)
    input('press any key to exit')
