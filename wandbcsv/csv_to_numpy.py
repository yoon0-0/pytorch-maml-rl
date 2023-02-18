from numpy import loadtxt
import numpy as np
# converters = {0: lambda s: float(s.strip('"'))}


return_path = "train/halfcheetah_legweight_return.csv"
xdiff_path = "train/halfcheetah_legweight_xdiff.csv"

# Count num of columns
with open(return_path) as f:
    ncols = len(f.readline().split(','))

return_data = loadtxt(return_path, delimiter=',', skiprows=1, usecols=range(1, ncols), dtype='str', unpack=True)
return_data = np.char.strip(return_data, '"')
print(return_data.shape)
return_data = return_data.astype(np.float)

xdiff_data = loadtxt(xdiff_path, delimiter=',', skiprows=1, usecols=range(1, ncols), dtype='str', unpack=True)
xdiff_data = np.char.strip(xdiff_data, '"')
xdiff_data = xdiff_data.astype(np.float)

# print(xdiff_data.shape)
np.savez("halfcheetah_legweight_train", x_diff=xdiff_data[[0],:], reward=return_data[[0],:])