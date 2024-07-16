import numpy as np

a = [2, 4, 8, 16, 32]
temp = []
for i in a:
    temp = temp + [np.load("output_dim_pure_nInE" + str(i) + ".npy")]
np.save("output_bayesrisk_dimcomp_pure_240711", np.array(temp))