from .FSRCNN import FSRCNN
from .SRCNN import SRCNN
from .VDSR import VDSR
from .ESPCN import ESPCN, espcn_x4
from .IDN import IDN
from .EDSR import EDSR
from .Interpolate import Interpolate

model_names = ['FSRCNN', 'SRCNN', 'VDSR', 'EDSR', 'ESPCN', 'IDN', 'EDSR', 'Interpolate']
__all__ = model_names + ['espcn_x4']