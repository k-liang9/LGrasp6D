import os
import zipfile
import numpy as np
import multiprocessing as mp
from multiprocessing import Pool

dataset_path = 'data/Grasp-Anything-6D'

def add_one(num):
    return num+1

if __name__ == '__main__':
    with Pool(processes=len(os.sched_getaffinity(0))) as pool:
        results = pool.map(add_one, range(4))
        print(results)