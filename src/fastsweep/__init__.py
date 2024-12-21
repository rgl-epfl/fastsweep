import drjit as dr

from _fastsweep_core import __doc__, __version__
from _fastsweep_core import redistance as _redistance

def redistance(data: dr.llvm.TensorXf | dr.cuda.TensorXf) -> dr.llvm.TensorXf | dr.cuda.TensorXf:
    return _redistance(dr.detach(data, preserve_type=False))
