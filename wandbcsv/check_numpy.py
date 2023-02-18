import numpy as np

halfcheetah_boxweight_3 = np.load("halfcheetah_boxweight_3.npz")
# for i in halfcheetah_boxweight_3:
#     print(i)

print(halfcheetah_boxweight_3['x_diff'].shape)