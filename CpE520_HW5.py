print(__doc__)

from time import time
import numpy as np
import matplotlib.pyplot as plt
import h5py

from sklearn import metrics
from sklearn.cluster import KMeans
from sklearn.datasets import load_digits
from sklearn import decomposition
from sklearn.preprocessing import scale
from skimage.transform import rescale

import readMnist
#temp = mnist.read('train', '/home/crob/Downloads/mnist/')


f = h5py.File('mnist_train.h5','r')
mnist_train = f['data'][:]
mnist_train_labels = f['labels'][:]
f.close()
f = h5py.File('mnist_test.h5','r')
mnist_test = f['data'][:]
mnist_test_labels = f['labels'][:]
f.close()

#im = np.ndarray
#for img in temp:
#   im.append(img)


np.random.seed(69)

digits = load_digits()
data = scale(digits.data)

#n_samples, n_features = data.shape
n_samples = len(mnist_train)
n_features = np.size(mnist_train[1])

#n_digits = len(np.unique(digits.target))
n_digits = 10
#labels = digits.target

sample_size = 300

print("n_digits: %d, \t n_samples %d, \t n_features %d"
      % (n_digits, n_samples, n_features))


print(79 * '_')
print('% 9s' % 'init'
      '    time  inertia    homo   compl  v-meas     ARI AMI  silhouette')



# Insert code for PCA
n_comp=5
pca = decomposition.PCA(n_components=n_comp)
out = pca.fit(mnist_train)
xtrain = pca.transform(mnist_train)
b = pca.components_.reshape((n_comp, 28, 28))
print(np.shape(b))






# estimator = KMeans(init='k-means++', n_clusters=n_digits, n_init=10) #name="k-means++", data=data)
# t0 = time()
# estimator.fit(temp)
# print('Time to complete %i seconds' % (time() - t0))
# print(estimator.labels_)
# print(len(estimator.cluster_centers_))
# a = estimator.cluster_centers_[1,:]
# b = np.reshape(estimator.cluster_centers_,(n_digits,28,28))
# #cent = np.reshape(a,(28,28))
# #imgplot = plt.imshow(cent)
# np.shape(b)

for i in range(len(b)):
    plt.subplot(5,len(b)/5,i)
    plt.tick_params(
        axis='both',
        which='both',
        bottom='off',
        top='off',
        left='off',
        right='off',
        labelbottom='off',
        labelleft='off')
    imgp=plt.imshow(b[i,:,:])
    plt.set_cmap('gray')
plt.suptitle('%i KMeans Centroids of MNIST Dataset' % len(b))
plt.show()

example = mnist_test[range(5),:]
ex = pca.inverse_transform(example)
print(np.shape(ex))
example = ex.components_.reshape(28,28)
print(np.shape(example))


for i in range(len(b)):
    plt.subplot(5,len(b)/5,i)
    plt.tick_params(
        axis='both',
        which='both',
        bottom='off',
        top='off',
        left='off',
        right='off',
        labelbottom='off',
        labelleft='off')
    c = np.dot(b[i,:,:],example)
    imgp=plt.imshow(c)
    plt.set_cmap('gray')
plt.suptitle('%i KMeans Centroids of MNIST Dataset' % len(b))
plt.show()
