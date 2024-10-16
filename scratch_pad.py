'''
Some bits of code taken from https://github.com/yosukekatada/Hopfield_network
variable names are inspired from the Artem Kirsanov's video on Hopfield Network.
'''

#%%
import os
import numpy as np
from tqdm import tqdm

from glob import glob
import matplotlib.pyplot as plt
from src.hopfield_network import BipolarHN, BinaryHN, plot

data_path = 'D:\KJ_personal\Stuff\OCL\hopfield_network\Hopfield_network-master'

#%%

img_shape = (100, 100)
img_paths = [
    os.path.join(data_path, 'train_pics\\1.jpg'),
    os.path.join(data_path, 'train_pics\\2.jpg'),
]
cue_img_paths = [
    os.path.join(data_path, 'test_pics\\1_1.jpg'),
    # os.path.join(data_path, 'test_pics\\2_1.jpeg'),
]

cue_img_paths = glob(os.path.join(data_path, 'test_pics\\*.jpg'))

#%%

################################# BIPOLAR HN #######################################

model = BipolarHN(img_shape, img_paths, threshold=65, thinking_time=40_000)
res = model.remember(cue_img_paths)

mem_cues, recalled_memories = [], []
for k,v in res.items():
    mem_cues.append(v['mem_cue'])
    recalled_memories.append(v['recalled_mem'])

plot(model.memories)
plot(mem_cues)
plot(recalled_memories)

for k,v in res.items():
    print(v['match_score'], end=', ')
print()
for k,v in res.items():
    print(v['energies'][-1], end=', ') 


#%%
################################# BINARY HN #######################################

model = BinaryHN(img_shape, img_paths, threshold=65, thinking_time=40_000)
res = model.remember(cue_img_paths)

mem_cues, recalled_memories = [], []
for k,v in res.items():
    mem_cues.append(v['mem_cue'])
    recalled_memories.append(v['recalled_mem'])

plot(model.memories)
plot(mem_cues)
plot(recalled_memories)

for k,v in res.items():
    print(v['match_score'], end=', ')
print()
for k,v in res.items():
    print(v['energies'][-1], end=', ') 

#%%
for k,v in res.items():
    plt.plot(v['energies'])
    plt.show()


#%%

model_bp = BipolarHN(img_shape, img_paths, threshold=65, thinking_time=40_000)
model_bn = BinaryHN(img_shape, img_paths, threshold=65, thinking_time=40_000)
