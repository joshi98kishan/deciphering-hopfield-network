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

class HopfieldNetwork:
    def __init__(self, img_shape: tuple, mem_img_paths: list, threshold: int, thinking_time: int):
        self.img_shape = img_shape
        self.mem_img_paths = mem_img_paths
        self.threshold = threshold
        self.thinking_time = thinking_time
        
        self.num_nodes = img_shape[0]*img_shape[1]
        self.W_shape = (self.num_nodes, self.num_nodes)
        self.W = np.zeros(self.W_shape)

        self.memories = self.fit_and_get_memories()


    def read_img_to_mem(self, img_path):
        pil_img = Image.open(img_path).convert(mode="L")
        pil_img = pil_img.resize(self.img_shape)
        mem = np.array(pil_img, dtype = float)
        mask = mem > self.threshold
        mem[mask] = 1
        mem[~mask] = -1
        return mem
    
    def mem_to_img(self, mem):
        img = np.zeros(self.img_shape, dtype=np.uint8)
        img[mem==1] = 255
        img[mem==-1] = 0
        img = Image.fromarray(img, mode="L")
        return img

    def fit_and_get_memories(self):
        memories = []
        for path in self.mem_img_paths:
            mem = self.read_img_to_mem(path)
            memories.append(mem)

        #####################
        self.fit(memories)
        #####################

        return memories
    
    def fit(self, memories):
        for mem in memories:
            self.W += self.calc_mem_weights(mem)

    def calc_mem_weights(self, mem):
        x = mem.flatten()
        mem_w = np.zeros(self.W_shape)
        for i in tqdm(range(len(x))):
            for j in range(i+1, len(x)):
                mem_w[i,j] = x[i]*x[j]
                mem_w[j,i] = mem_w[i,j]
        return mem_w
        
    def remember_(self, mem_cue):
        x = mem_cue.flatten()
        self.energies_ = []

        for t in tqdm(range(self.thinking_time)):
            i = random.randint(0, len(x)-1)
            i_local_field = np.dot(self.W[i][:], x)

            if i_local_field > 0:
                x[i] = 1
            elif i_local_field < 0:
                x[i] = -1

            ##########################
            if t%1000 == 0:
                self.energies_.append(self.calc_energy(x))

        recalled_mem = x.reshape(self.img_shape)
        return recalled_mem

    def remember(self, memcue_img_paths):
        recalled_memories = []
        self.mem_cues = []
        self.memcue_img_paths = memcue_img_paths

        for path in memcue_img_paths:
            mem_cue = self.read_img_to_mem(path)
            self.mem_cues.append(mem_cue)

            recalled_mem = self.remember_(mem_cue)
            recalled_memories.append(recalled_mem)

        ###########################
        self.recalled_memories = recalled_memories
        
        return recalled_memories
    
    def calc_energy(self, act_vec):
        energy = -(act_vec@(self.W@act_vec))
        return energy

    def match_memories(self, mem1, mem2):
        return (mem1==mem2).sum()/self.num_nodes

    def evaluate(self):
        scores = []
        for memcue_img_path, rec_mem in zip(self.memcue_img_paths, self.recalled_memories):
            _, fname_with_ext = os.path.split(memcue_img_path)
            fname = fname_with_ext[:-4]
            corres_mem_img_fname = fname.split('_')[0]
            corres_mem = self.memories[int(corres_mem_img_fname)-1]

            scores.append(self.match_memories(rec_mem, corres_mem))

        return np.mean(scores), np.std(scores)

