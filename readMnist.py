__author__ = 'Christo Robison'

import numpy as np
import scipy as sci
import h5py
from PIL import Image
import os
import collections

'''THis program reads in the entire MNist dataset'''


def readMnist(path = '/home/crob/Downloads/mnist/train'):
    dataDir = '/home/crob/Downloads/mnist/train/'
    #m = np.ndarray()
    a = []
    l = []
    for root, dirs, files in os.walk(path):
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
                l.append(root.replace(path+'/',''))
                #print(root.replace(path+'/',''))

    print(len(a))
    p = np.array(a)
    print(p.shape)
    out = collections.namedtuple('examples',['data', 'label'])
    o = out(data=p, label=l)
    return o

if __name__ == '__main__':
    path = '/home/crob/Downloads/mnist'
    s = readMnist(path+'/test')
    print(np.shape(s.data))
    f = h5py.File("mnist_test.h5","w")
    f.create_dataset('data', data=s.data)
    f.create_dataset('labels', data=s.label)
    f.close()
