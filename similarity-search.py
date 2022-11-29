import numpy as np
from pynndescent import NNDescent

# load the file with all the 2048 features of all the bounding boxes
data = np.load('features.npy')

index = NNDescent(data)
# find similar boxes for box 58
neighbors = index.query(data[58:59,:], k=11)
print(f'Box 58 neighbors: {neighbors[0][0][1:]}')
# find similar boxes for box 24
neighbors = index.query(data[24:25,:], k=11)
print(f'Box 24 neighbors: {neighbors[0][0][1:]}')
# find similar boxes for box 27
neighbors = index.query(data[27:28,:], k=11)
print(f'Box 27 neighbors: {neighbors[0][0][1:]}')
# find similar boxes for box 30
neighbors = index.query(data[30:31,:], k=11)
print(f'Box 30 neighbors: {neighbors[0][0][1:]}')
# find similar boxes for box 31
neighbors = index.query(data[31:32,:], k=11)
print(f'Box 31 neighbors: {neighbors[0][0][1:]}')