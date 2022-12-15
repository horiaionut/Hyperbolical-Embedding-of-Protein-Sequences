import numpy as np
import random

def choice(lists, samples_per_list=1):
    result = np.empty((samples_per_list, len(lists)))

    for idx, l in enumerate(lists):
        result[:, idx] = np.random.choice(l, samples_per_list)
    
    return result