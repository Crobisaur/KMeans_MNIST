__author__ = 'Christo Robison'

import numpy as np
import scipy as sci
from PIL import Image
import os

'''THis program reads in the entire MNist dataset'''


def readMnist(path = '/home/crob/Downloads/mnist/train'):
    dataDir = '/home/crob/Downloads/mnist/train/'
    #m = np.ndarray()
    a = []
    for root, dirs, files in os.walk(dataDir):
        for name in files:
            if name.endswith(".png"):
                #a = sci.ndimage.imread(name, True)
                #print(root, name)
                im = np.array(Image.open(os.path.join(root, name)).convert('L'), 'f')
                #print(im.shape)
                np.hstack(im)
                #print(im.shape)
                #print(im)
                a.append(np.reshape(im, 784))

    print(len(a))
    p = np.array(a)
    print(p.shape)
    return p

if __name__ == '__main__':
    s = readMnist()