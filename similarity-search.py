import numpy as np
from pynndescent import NNDescent

data = np.load('features.npy')

index = NNDescent(data)
neighbors = index.query(data[58:59,:], k=11)
print(f'Box 58 neighbors: {neighbors[0][0][1:]}')
neighbors = index.query(data[24:25,:], k=11)
print(f'Box 24 neighbors: {neighbors[0][0][1:]}')
neighbors = index.query(data[27:28,:], k=11)
print(f'Box 27 neighbors: {neighbors[0][0][1:]}')
neighbors = index.query(data[30:31,:], k=11)
print(f'Box 30 neighbors: {neighbors[0][0][1:]}')
neighbors = index.query(data[31:32,:], k=11)
print(f'Box 31 neighbors: {neighbors[0][0][1:]}')