import numpy as np
from phidl import Device
from phidl import Group
import phidl.geometry as pg
import phidl.routing as pr
import phidl.path as pp
from phidl import quickplot as qp
from phidl import set_quickplot_options
#set_quickplot_options(blocking=True, show_ports=True)
set_quickplot_options(blocking=True, show_ports=False)

mu0 = 1.257e-6 # N/A^2
lambda_NbN = 390e-9 # nm
t_NbN = 16e-9
W_NbN = 500e-9

#import qnngds.utilities as qu
#import qnngds.geometry as qg

inner_loop = np.genfromtxt('inner_loop.csv', delimiter=',')
left_loop = np.genfromtxt('left_loop.csv', delimiter=',')
right_loop = np.genfromtxt('right_loop.csv', delimiter=',')

D = Device("test")

inner = Device("inner")
inner.add_polygon(inner_loop)
left = Device("left")
left.add_polygon(left_loop)
right = Device("right")
right.add_polygon(right_loop)

D << inner
D << left
D << right

area_inner = inner.area()
length_inner = sum((np.diff(inner_loop[:,0])**2 + np.diff(inner_loop[:,1])**2)**0.5)
print(f'area_inner = {round(area_inner)}um^2')
print(f'length_inner = {round(length_inner)}um');
print(f'area_inner/length_inner = {round(area_inner/length_inner*1e3)}nm')

area_left = left.area()
length_left = sum((np.diff(left_loop[:,0])**2 + np.diff(left_loop[:,1])**2)**0.5)
print(f'area_left = {round(area_left)}um^2')
print(f'length_left = {round(length_left)}um');
print(f'area_left/length_left = {round(area_left/length_left*1e3)}nm')

area_right = right.area()
length_right = sum((np.diff(right_loop[:,0])**2 + np.diff(right_loop[:,1])**2)**0.5)
print(f'area_right = {round(area_right)}um^2')
print(f'length_right = {round(length_right)}um');
print(f'area_right/length_right = {round(area_right/length_right*1e3)}nm')

I_inner = t_NbN*W_NbN/(lambda_NbN**2)*area_inner/length_inner*1e-3
I_left = -t_NbN*W_NbN/(lambda_NbN**2)*area_left/length_left*1e-3
I_right = -t_NbN*W_NbN/(lambda_NbN**2)*area_right/length_right*1e-3
I_tot = I_inner + I_left + I_right
print(f'I_tot = {round(I_tot*1e6)}uA = I_inner + I_left + I_right = {round(I_inner*1e6)}uA + {round(I_left*1e6)}uA + {round(I_right*1e6)}uA')
qp(D)
