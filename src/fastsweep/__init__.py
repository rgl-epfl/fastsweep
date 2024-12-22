import drjit as dr

from .fastsweep_ext import __doc__, __version__
from .fastsweep_ext import redistance as _redistance

def redistance(data):
    data = dr.detach(data, preserve_type=False)
    return _redistance(data)
