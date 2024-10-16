import os
import random
import numpy as np
from PIL import Image
from tqdm import tqdm

from math import ceil
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
# %%

def plot(imgs):
    n = len(imgs)
    if n>=5:
        ncols = 5
    else:
        ncols = n
    nrows = ceil(n/5)

    fig = plt.figure(figsize=(2*ncols, 2*nrows))
    grid = ImageGrid(fig, 111,  # similar to subplot(111)
                    nrows_ncols=(nrows, ncols),  
                    axes_pad=0.1,  # pad between Axes in inch.
                    )
    
    for ax, im in zip(grid, imgs):
        ax.imshow(im, cmap='gray')

    plt.show()

class BaseHopfieldNetwork:
    def __init__(self, num_nodes, thinking_time):
        self.OFF_ACT_VAL: int
        self.results = {}
        self.stopping_step = []
        self.rng = np.random.default_rng()

        self.thinking_time = thinking_time
        self.num_nodes = num_nodes
        
        W_shape = (self.num_nodes, self.num_nodes)
        self.W = np.zeros(W_shape)

    def fit_and_get_memories(self):
        '''
        create memories list, calls `fit` on it and returns that list
        '''
        pass

    def fit(self, memories):
        for mem in memories:
            self.W += self.calc_mem_weights(mem)

    def calc_mem_weights(self):
        pass

    def remember_(self, mem):
        mem_prev = mem.copy()
        mem_energies = []
        idxs_ = np.arange(len(mem))
        print('>> Sequential Remembering')
        for t in tqdm(range(self.thinking_time)):
            self.rng.shuffle(idxs_)
            for i in idxs_:
                i_local_field = np.dot(self.W[i][:], mem)

                if i_local_field > 0:
                    mem[i] = 1
                elif i_local_field < 0:
                    mem[i] = self.OFF_ACT_VAL

                ##########################
                if i%4000 == 0:
                    mem_energies.append(self.calc_energy(mem))

            stop = np.equal(mem, mem_prev).all()
            if stop:
                # print('stopping at', t)
                self.stopping_step.append(t)
                break
            else:
                mem_prev = mem.copy()

        else:
            self.stopping_step.append(t)

        return mem, mem_energies
    

    def remember(self):
        '''
        calls `remember_` for each mem cue, fill the `results` dict and returns it.
        '''
        
    
    def calc_energy(self, mem):
        energy = -(mem@(self.W@mem))
        return energy

    def match_memories(self, mem1, mem2):
        return (mem1==mem2).sum()/self.num_nodes


class BaseImageHopfieldNetwork(BaseHopfieldNetwork):
    def __init__(self, img_shape: tuple, mem_img_paths: list, threshold: int, thinking_time):
        super().__init__(img_shape[0]*img_shape[1], thinking_time)

        self.img_shape = img_shape
        self.mem_img_paths = mem_img_paths
        self.threshold = threshold

    def read_img_to_mem(self, img_path):
        pil_img = Image.open(img_path).convert(mode="L")
        pil_img = pil_img.resize(self.img_shape)
        mem = np.array(pil_img, dtype = float)
        mask = mem > self.threshold
        mem[mask] = 1
        mem[~mask] = self.OFF_ACT_VAL
        mem = mem.flatten()
        return mem
    
    # def mem_to_img(self, mem):
    #     img = np.zeros(self.img_shape, dtype=np.uint8)
    #     img[mem==1] = 255
    #     img[mem==self.OFF_ACT_VAL] = 0
    #     img = Image.fromarray(img, mode="L")
    #     return img
    
    def fit_and_get_memories(self):
        memories = []
        for path in self.mem_img_paths:
            mem = self.read_img_to_mem(path)
            memories.append(mem)

        #####################
        self.fit(memories)
        #####################

        return memories
    
    
    def extract_fname(self, path):
        _, fname_with_ext = os.path.split(path)
        fname = fname_with_ext[:-4]
        return fname

    def remember(self, memcue_img_paths):
        '''
        Test images should be corresponding to the train images
        Name of a test image should start with the corresponding train image
        '''

        for path in memcue_img_paths:
            res = {}

            mem_cue = self.read_img_to_mem(path)
            recalled_mem, res['energies'] = self.remember_(mem_cue)
            res['mem_cue_img'] = mem_cue.reshape(self.img_shape)
            res['recalled_mem_img'] = recalled_mem.reshape(self.img_shape)
            
            fname = self.extract_fname(path)
            corres_mem_img_fname = fname.split('_')[0]
            corres_mem = self.memories[int(corres_mem_img_fname)-1]
            match_score = self.match_memories(recalled_mem, corres_mem)
            res['match_score'] = match_score
            
            self.results[fname] = res
        
        return self.results


class ImBipolarHN(BaseImageHopfieldNetwork):
    def __init__(self, img_shape: tuple, mem_img_paths: list, threshold: int, thinking_time: int):
        super().__init__(img_shape, mem_img_paths, threshold, thinking_time)
        self.OFF_ACT_VAL = -1
        
        self.memories = self.fit_and_get_memories()

    def calc_mem_weights(self, mem):
        mem_w = mem.reshape((-1, 1))@mem.reshape((1, -1))
        np.fill_diagonal(mem_w, 0)
        return mem_w
    

class ImBinaryHN(BaseImageHopfieldNetwork):
    def __init__(self, img_shape: tuple, mem_img_paths: list, threshold: int, thinking_time: int):
        super().__init__(img_shape, mem_img_paths, threshold, thinking_time)
        self.OFF_ACT_VAL = 0

        self.memories = self.fit_and_get_memories()

    def calc_mem_weights(self, mem):
        x = 2*mem - 1
        mem_w = x.reshape((-1, 1))@x.reshape((1, -1))
        np.fill_diagonal(mem_w, 0)
        return mem_w
