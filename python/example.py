"""This script shows how the library can be used to convert a generic implicit function to an SDF"""

import matplotlib.pyplot as plt
import numpy as np

# Using CUDA will be *much* faster than running on CPU, but requires a CUDA capable GPU
use_cuda = False

if not use_cuda:
    from drjit.llvm import TensorXf
else:
    from drjit.cuda import TensorXf

import fastsweep

res = 64

# Create an initial implicit function that defines a sphere (but is not a valid SDF)
z, y, x = np.meshgrid(*[np.linspace(0, 1, res)] * 3, indexing='ij')
pts = np.stack([x.ravel(), y.ravel(), z.ravel()], axis=1)
mask = ((np.linalg.norm(pts - 0.5, axis=-1) - 0.3) > 0.0).astype(np.float32)
# Make sure the inside is negative and the outside is positive
init = TensorXf(np.reshape(mask - 0.5, (res, res, res)))

# Re-distance to obtain an SDF
sdf = fastsweep.redistance(init)

# Plot the results
fig, ax = plt.subplots(1, 2)
ax[0].imshow(init[:, res // 2, :], cmap='coolwarm_r')
ax[0].set_title("Initialization")
ax[1].imshow(sdf[:, res // 2, :], cmap='coolwarm_r')
x, y = np.meshgrid(np.arange(res), np.arange(res))
ax[1].contour(x, y, sdf[:, res // 2, :], levels=[0], colors='red')
ax[1].set_title("SDF")
plt.show()
