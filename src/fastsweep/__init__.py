import drjit as _dr

from ._fastsweep_core import __doc__, __version__
from ._fastsweep_core import redistance as _redistance

def redistance(data, dx=None):
    data = _dr.detach(data, preserve_type=False)
    return _redistance(data)
