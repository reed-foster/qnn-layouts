import phidl.geometry as pg
from phidl import quickplot as qp
from phidl import set_quickplot_options
set_quickplot_options(blocking=True)

D = pg.compass(size = (4,2), layer = 0)
qp(D) # quickplot the geometry
