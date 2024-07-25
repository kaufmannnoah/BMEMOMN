import numpy as np

a = np.load("out_dim4_B.npy")
b = np.load("out_dim4_GP.npy")

out = np.zeros((5, 3, 1, 4, 12, 4000))
out[:, 0] = a[:, :, 0]

temp = np.zeros((5, 2, 1, 4, 12, 4000))
temp[:4] = b

out[:, 1:] = temp

np.save("out_dim4_BGP", out)