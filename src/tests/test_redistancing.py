import pytest
import numpy as np
import drjit as dr
import fastsweep


def create_sphere_sdf(res, center=(0.5, 0.5, 0.5), radius=0.3):
    import numpy as np
    z, y, x = np.meshgrid(*[np.linspace(0, 1, res[i]) for i in range(3)], indexing='ij')
    pts = np.stack([x.ravel(), y.ravel(), z.ravel()], axis=1)
    signed_dist = np.linalg.norm(pts - center, axis=-1) - radius
    sdf = np.reshape(signed_dist, res)
    return sdf.astype(np.float32)


def test01_cpu():
    res = (128, 128, 128)
    sdf = create_sphere_sdf(res)
    sdf = dr.llvm.TensorXf(sdf)
    sdf_redistanced = fastsweep.redistance(sdf)

    diff = dr.max(dr.abs(sdf - sdf_redistanced))
    assert np.array(diff) < 0.02


@pytest.mark.skipif(not dr.has_backend(dr.JitBackend.CUDA), reason="No CUDA device available")
def test02_gpu():
    res = (128, 128, 128)
    sdf = create_sphere_sdf(res)
    sdf = dr.cuda.TensorXf(sdf)
    sdf_redistanced = fastsweep.redistance(sdf)

    diff = dr.max(dr.abs(sdf - sdf_redistanced))
    assert np.array(diff) < 0.02

@pytest.mark.skipif(not dr.has_backend(dr.JitBackend.CUDA), reason="No CUDA device available")
def test03_gpu_vs_gpu():
    res = (64, 64, 64)
    sdf = create_sphere_sdf(res)
    sdf_redistanced_cuda = fastsweep.redistance(dr.cuda.TensorXf(sdf))
    sdf_redistanced_cpu = fastsweep.redistance(dr.llvm.TensorXf(sdf))
    diff = np.max(np.abs(np.array(sdf_redistanced_cpu) - np.array(sdf_redistanced_cuda)))

    # The two different implementations are expected to match almost perfectly
    assert diff < 1e-7
