import numpy as np

with np.load('scada1.npz') as data:
    for item in data.files:
        print(item)
