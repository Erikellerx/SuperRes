from .FSRCNN import FSRCNN
from .SRCNN import SRCNN
from .VDSR import VDSR
from .ESPCN import ESPCN
from .IDN import IDN
from .EDSR import EDSR

model_names = ['FSRCNN', 'SRCNN', 'VDSR', 'EDSR', 'ESPCN', 'IDN', 'EDSR']
__all__ = model_names