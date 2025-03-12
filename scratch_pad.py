'''
Some bits of code taken from https://github.com/yosukekatada/Hopfield_network
variable names are inspired from the Artem Kirsanov's video on Hopfield Network.
'''

#%%
import os
import numpy as np
from tqdm import tqdm
from collections import defaultdict

from glob import glob
import matplotlib.pyplot as plt
from src.hopfield_network import plot, ImageHopfieldNetwork, VecHopfieldNetwork

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

model = ImageHopfieldNetwork(img_shape, img_paths, threshold=65, thinking_time=40_000, off_act_val=-1)
res = model.remember(cue_img_paths)

mem_cues, recalled_memories = [], []
for k,v in res.items():
    mem_cues.append(v['mem_cue_img'])
    recalled_memories.append(v['recalled_mem_img'])

plot([mem.reshape(img_shape) for mem in model.memories])
plot(mem_cues)
plot(recalled_memories)

for k,v in res.items():
    print(v['match_score'], end=', ')
print()
for k,v in res.items():
    print(v['energies'][-1], end=', ') 


#%%
################################# BINARY HN #######################################

model = ImageHopfieldNetwork(img_shape, img_paths, threshold=65, thinking_time=40_000, off_act_val=0)
res = model.remember(cue_img_paths)

mem_cues, recalled_memories = [], []
for k,v in res.items():
    mem_cues.append(v['mem_cue_img'])
    recalled_memories.append(v['recalled_mem_img'])

plot([mem.reshape(img_shape) for mem in model.memories])
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
################################# XOR Distribution #######################################
def show_res(res, model, show_vis_dist=False):
    dist = defaultdict(int)
    vis_dist = defaultdict(int)
    for v in res.values():
        dist[tuple(v['recalled_mem'])] += 1
        vis_dist[tuple(v['recalled_mem'][:3])] += 1

    for k in sorted(dist.keys()):
        print('{:10} | {:^10} | {:^10}'.format(str(k), dist[k], model.calc_energy(np.array(k))))
    
    if show_vis_dist:
        for k in sorted(vis_dist.keys()):
            print('{:10} | {:^10}'.format(str(k), vis_dist[k]))

#%%

NUM_NODES = 3
THINKING_TIME = 100
off_act_val = o = 0

# only visible units
memories = np.array([
    [o, o, o],
    [o, 1, 1],
    [1, o, 1],
    [1, 1, o]
])


# with hidden units
NUM_NODES = 5
memories = np.array([
    [o, o, o, o, o],
    [o, 1, 1, 1, o],
    [1, o, 1, o, 1],
    [1, 1, o, o, o]
])

model = VecHopfieldNetwork(memories, THINKING_TIME, off_act_val)
model.W
#%%

# testing
mem_cues = np.array([
    [o, o, o, o, o],
    [o, 1, 1, o, o],
    [1, o, 1, o, o],
    [1, 1, o, o, o]
])

res = model.remember(mem_cues)
show_res(res, model, show_vis_dist=True)

#%%
# testing with random init states
pop_size = 10_000
rng = np.random.default_rng()
init_states = rng.integers(0, 2, size=(pop_size, NUM_NODES))

res = model.remember(init_states)
show_res(res, model, show_vis_dist=True)

#%%

# testing with random init states having 0 hidden unit value
pop_size = 10_000
rng = np.random.default_rng()
init_states = rng.integers(0, 2, size=(pop_size, NUM_NODES))
init_states[:, 3:] = 0

res = model.remember(init_states)
show_res(res, model, show_vis_dist=True)

#%%
# testing with random init states having 0 hidden unit value and output unit value

pop_size = 10_000
rng = np.random.default_rng()
init_states = rng.integers(0, 2, size=(pop_size, NUM_NODES))
init_states[:, 2:] = 0

res = model.remember(init_states)
show_res(res, model, show_vis_dist=True)

#%%

################################# Shifter Problem ######################
def gen_shifter_data(num_bits, pop_size):
    rng = np.random.default_rng()
    rand_part = rng.integers(0, 2, size=(pop_size, num_bits))
    init_states = []
    for rpart in rand_part:
        label_idx = rng.integers(0, 3)
        shift_part = np.roll(rpart, label_idx-1)
        arr_ = np.append(rpart, shift_part)
        label_arr = [0, 0, 0]
        label_arr[label_idx] = 1
        arr_ = np.append(arr_, label_arr)
        init_states.append(arr_)
    init_states = np.array(init_states)
    return init_states

num_bits = 8
pop_size = 10_000
THINKING_TIME = 100
off_act_val = 0
memories = gen_shifter_data(num_bits, pop_size)
num_nodes = 2*num_bits+3
model = VecHopfieldNetwork(memories, THINKING_TIME, off_act_val)
W = model.W
plt.matshow(model.W)

#%%
import itertools 

def gen_complete_shifter_data(num_bits):
    rand_part = np.array(list(itertools.product([0, 1], repeat=num_bits)))
    init_states = []
    for rpart in rand_part:
        for label_idx in range(3): 
            shift_part = np.roll(rpart, label_idx-1)
            arr_ = np.append(rpart, shift_part)
            label_arr = [0, 0, 0]
            label_arr[label_idx] = 1
            arr_ = np.append(arr_, label_arr)
            init_states.append(arr_)

    init_states = np.array(init_states)
    return init_states
#%%

num_bits = 8
THINKING_TIME = 100
off_act_val = 0
memories = gen_complete_shifter_data(num_bits)
num_nodes = 2*num_bits+3
model = VecHopfieldNetwork(num_nodes, THINKING_TIME, memories, off_act_val)
W = model.W
plt.matshow(model.W)

#%%
################################# Parity Problem ######################

