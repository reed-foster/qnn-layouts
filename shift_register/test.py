import numpy as np
from phidl import Device
from phidl import Group
from phidl import Path
import phidl.geometry as pg
import phidl.routing as pr

from phidl import quickplot as qp
from phidl import set_quickplot_options
set_quickplot_options(blocking=True, show_ports=True)

p = Path([(0,0), (0,1), (1,1), (1,2)])
qp(p)
