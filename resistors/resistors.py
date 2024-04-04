# -*- coding: utf-8 -*-
"""
Created on Wed Apr 03 17:00:03 2024

@author: reedf

"""
import numpy as np
from phidl import Device
import phidl.geometry as pg

from phidl import quickplot as qp
from phidl import set_quickplot_options
#set_quickplot_options(blocking=True, show_ports=True)
set_quickplot_options(blocking=False, show_ports=True, new_window=True)

def resistor_meander(layer=1,
                     width=2,
                     pitch=4,
                     squares=100,
                     max_length=20,
                     aspect_ratio=1,
                     ):
    """
    Create resistor meander with specified number of squares
    
    Parameters:
        layer (Int): GDS layer
        width (Float): width in microns
        pitch (Float): desired pitch of meander in microns
        squares (Float): desired number of squares
        max_length (Float): desired length of device
        aspect_ratio (Float): desired w/h ratio of meander

    Returns:
        D (Device)
    """
    D = Device('resistor_meander')

    meander_spacing = (pitch - width)/width

    if width*squares < max_length:
        # just make a straight
        return pg.straight(size=(width, width*squares), layer=layer)

    # make meander
    def hairpin(hp_length):
        """Create hairpin used in meander"""
        H = Device('hairpin')
        straight = pg.rectangle(size=(hp_length - width, width), layer=layer)
        conn = pg.rectangle(size=(width, (2 + meander_spacing)*width), layer=layer)
        for i in range(2):
            s = H << straight
            s.move((-s.xmax, -s.ymin + (1 + meander_spacing)*width*i))
        c = H << conn
        c.move((-c.xmin, -c.ymin))
        H = pg.union(H, by_layer=True)
        H.add_port(name=1, midpoint=(-hp_length + width, width/2), width=width, orientation=180)
        H.add_port(name=2, midpoint=(-hp_length + width, (1 + meander_spacing)*width + width/2), width=width, orientation=180)
        return H

    def stub(orientation):
        """Create stub to connect to meander ends"""
        S = Device('stub')
        straight = pg.rectangle(size=(width, 2*width), layer=layer)
        s = S << straight
        s.move((-s.x, -s.ymin))
        S.add_port(name=1, midpoint=(0, width/2), width=width, orientation=orientation)
        S.add_port(name=2, midpoint=(0, 2*width), width=width, orientation=90)
        return S

    # solve system for hp_length, n_turn given squares, pitch, aspect_ratio, width:
    # squares = 2 * (2 * hp_length/width + 0.5) * n_turn
    # width_meander = 2 * hp_length
    # height_meander = 2 * pitch * n_turn
    # width_meander/height_meander = aspect_ratio
    #
    # double number of hairpins
    squares -= 2 # account for extra squares from stubs
    n_turn = int(2*((16*aspect_ratio*squares*pitch*width+width**2)**0.5 - width)/(8*aspect_ratio*pitch))
    hp_length = (squares/n_turn-0.5)*width/2
    hp = hairpin(hp_length)
    hp_prev = None
    for i in range(n_turn):
        hp_i = D << hp
        if hp_prev is not None:
            hp_i.connect(hp_i.ports[2 - (i%2)], hp_prev.ports[2 - (i%2)])
        else:
            stub_top = D << stub(0)
            stub_top.connect(stub_top.ports[1], hp_i.ports[2])
        hp_prev = hp_i
    stub_bot = D << stub(180 * (n_turn % 2))
    stub_bot.connect(stub_bot.ports[1], hp_prev.ports[2 - (n_turn % 2)])
    D = pg.union(D, by_layer=True, precision=0.01)
    D.add_port(name=1, port=stub_top.ports[2])
    D.add_port(name=2, port=stub_bot.ports[2])

    return D

def resistor_negtone(width = 1,
                     squares = 2,
                     max_length = 50,
                     meander_pitch = 2,
                     contact_base = 2,
                     contact_height = 2,
                     outline_sup = 1,
                     routing = 1,
                     layer_res = 2,
                     layer_sup = 1,
                     ):
    D = Device('resistor')
    info = locals()
    aspect_ratio = (contact_base + 2 * outline_sup)/max_length * 1.5
    res = D << resistor_meander(layer=layer_res, width=width, pitch=max(meander_pitch, width + 1), squares=squares, max_length=max_length, aspect_ratio=aspect_ratio)
    stub = pg.straight(size=(width, outline_sup), layer=layer_res)
    contact = pg.straight(size=(contact_base,contact_height), layer=layer_res)
    contact_sup = pg.straight(size=(contact_base + 2*outline_sup, contact_height + 2*outline_sup), layer=layer_sup)
    rout = pg.straight(size=(routing, 2*routing), layer=layer_sup)
    ports = []
    for p, port in res.ports.items():
        s = D << stub
        s.connect(s.ports[1], port)
        c = D << contact
        c.connect(c.ports[1], s.ports[2])
        c_sup = D << contact_sup
        c_sup.center = c.center
        r = D << rout
        r.connect(r.ports[1], c_sup.ports[2-(p%2)])
        ports.append(r.ports[2])

    D = pg.union(D, by_layer=True)
    D.add_port(port=ports[0], name=1)
    D.add_port(port=ports[1], name=2)
    D.rotate(90)
    D.info = info
    return D

if __name__ == "__main__":
    import phidlfem.poisson as ps
    # simple unit test
    D = Device("test")
    r1 = resistor_negtone(width=1, squares=100, max_length=20, meander_pitch=2, contact_base=10, contact_height=2, outline_sup=1, routing=1)
    r3 = resistor_negtone(width=0.5, squares=2, contact_base=4, contact_height=2, outline_sup=1, routing=1)
    r4 = resistor_negtone(width=0.5, squares=7, contact_base=4, contact_height=2, outline_sup=1, routing=1)
    r4 = resistor_negtone(width=2, squares=20, max_length=15, meander_pitch=4, contact_base=8, contact_height=4, outline_sup=2, routing=2)
    D << r1
    D << r3
    D << r4
    #r5 = resistor_meander(layer=1, width=1, squares=200, max_length=25, pitch=2, aspect_ratio=2)
    #try:
    #    print(f'num_squares = {ps.get_squares(r5, 0.05)}')
    #except IndexError as e:
    #    print(f'oops: {e}')
    #ps.visualize_poisson(r5, 0.1)
    #D << r5

    D.distribute(direction = 'y', spacing = 10)
    qp(D)
    input('press any key to exit')
