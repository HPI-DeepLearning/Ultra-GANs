import glob
import keras
import scipy
import numpy as np
import os.path
import random
import imageio
import re

from config import *

non_decimal = re.compile(r'[^\d]+')

# List avaiable sequences
def prepare_data(directory):
    sequences = []
    dirs = glob.glob(directory)
    
    for i in range(len(dirs)):
        dir_name = dirs[i] + "/*"
        list = glob.glob(dir_name)
        list.sort(key=sortKeyFunc)
        sequences.append(list)
    
    return sequences
    
def sortKeyFunc(s):
    return int(os.path.basename(non_decimal.sub('', s))[-7:-4])
    
def open(path):
    image = scipy.misc.imread(path).astype(np.float)
    subimages = np.split(image / 255, input + output, axis=1)
    return [np.stack(augment(subimages[output:]), axis=-1) * 2 - 1, np.stack(subimages[:output], axis=-1)]
    
# Load image sequences
def load(sequence, subsequence_length):

    images = [open(seq) for seq in sequence]
    x = []
    y = []
    
    for i in range(len(sequence) - subsequence_length):
        local_x = [images[i+j][0] for j in range(subsequence_length)]
        local_y = [images[i+j][1] for j in range(subsequence_length)]
        x.append(np.stack([np.stack(local_x, axis=0)], axis=0))
        y.append(strip(np.stack([np.stack(local_y, axis=0)], axis=0)))
        
    return x, y
            
def augment(sequence):
    return [apply_contrast(apply_gaussian_noise(resize(s))) for s in sequence]
    
def resize(image):
    offset = 8

    h1 = int(np.ceil(np.random.uniform(1e-2, offset)))
    w1 = int(np.ceil(np.random.uniform(1e-2, offset)))
    
    out = np.zeros((size + 2*offset, size + 2*offset))
    out[offset:offset+size, offset:offset+size] = image
    return out[h1:h1+size, w1:w1+size]
    
def apply_contrast(image):
    # Apply random brightness but keep values in [0, 1]
    # We apply a quadratic function with the form y = ax^2 + bx
    # Visualization: https://www.desmos.com/calculator/zzz75gguna
    delta = random.uniform(-0.04, 0.04)
    a = -4 * delta
    b = 1 - a
    return a * (image*image) + b * (image)
    
def apply_gaussian_noise(image):
    # Apply gaussian noise but keep values in [0, 1]
    random_value = random.uniform(-0.01, 0.01)
    return np.clip(image + (random_value), 0.0, 1.0)

def strip(arr):
    return arr[:, sequence_crop:-sequence_crop]
    
def re_shape(arr):
    return np.reshape(arr, (1, sequence_total, size, size, output))
    
def save_image(input, gt, generated, path):
    all = np.concatenate((input, gt, generated), axis=4)
    all = np.squeeze(all)
    all = np.squeeze(np.concatenate(np.split(all, 4, axis=0), axis=1))
    all = np.squeeze(np.concatenate(np.split(all, 12, axis=2), axis=1))
    imageio.imwrite(path, (np.clip(all, 0.0, 1.0) * 255).astype(np.uint8))
    
def convert_image(image, size, border, rotation):
    image = np.rot90(image, 4 - rotation)
    
    temp = np.ones((size, size)) * 0.5
    temp[border[0] : border[1], border[2] : border[3]] = image
    
    image = np.reshape(temp, (size, size, 1))
    image = np.concatenate([image, image, image], axis=2)
    return (image * 255).astype(np.uint8)
