print(__doc__)

from time import time
import numpy as np
import matplotlib.pyplot as plt
import h5py
from sklearn import metrics
from sklearn.cluster import KMeans
from sklearn.datasets import load_digits
from sklearn.decomposition import PCA
from sklearn.preprocessing import scale
import readMnist
#temp = mnist.read('train', '/home/crob/Downloads/mnist/')
#temp = readMnist.readMnist()

f = h5py.File('mnist_NP.h5','r')
temp = f['mnist_numpy'][:]
f.close()

#im = np.ndarray
#for img in temp:
 #   im.append(img)

np.random.seed(69)

digits = load_digits()
data = scale(digits.data)

#n_samples, n_features = data.shape
n_samples = len(temp)
n_features = np.size(temp[1])

#n_digits = len(np.unique(digits.target))
n_digits = 40
#labels = digits.target

sample_size = 300

print("n_digits: %d, \t n_samples %d, \t n_features %d"
      % (n_digits, n_samples, n_features))


print(79 * '_')
print('% 9s' % 'init'
      '    time  inertia    homo   compl  v-meas     ARI AMI  silhouette')


def bench_k_means(estimator, name, data):
    t0 = time()
    estimator.fit(data)
    print('% 9s   %.2fs    %i   %.3f   %.3f   %.3f   %.3f   %.3f    %.3f'
          % (name, (time() - t0), estimator.inertia_,
             metrics.homogeneity_score(labels, estimator.labels_),
             metrics.completeness_score(labels, estimator.labels_),
             metrics.v_measure_score(labels, estimator.labels_),
             metrics.adjusted_rand_score(labels, estimator.labels_),
             metrics.adjusted_mutual_info_score(labels,  estimator.labels_),
             metrics.silhouette_score(data, estimator.labels_,
                                      metric='euclidean',
                                      sample_size=sample_size)))
    print(estimator.labels_)
    print(len(estimator.cluster_centers_))
    a = estimator.cluster_centers_[1,:]
    np.shape(a)
    imgplot = plt.imshow(a)

estimator = KMeans(init='k-means++', n_clusters=n_digits, n_init=10) #name="k-means++", data=data)
t0 = time()
estimator.fit(temp)
print('Time to complete %i seconds' % (time() - t0))
print(estimator.labels_)
print(len(estimator.cluster_centers_))
a = estimator.cluster_centers_[1,:]
b = np.reshape(estimator.cluster_centers_,(n_digits,28,28))
#cent = np.reshape(a,(28,28))
#imgplot = plt.imshow(cent)
np.shape(b)
if n_digits == 40:
    outf = h5py.File('mnist40_Centroids.h5','w')
    outf.create_dataset('mnist40_Centroids_numpy', data=estimator.cluster_centers_)
    outf.close()

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

#bench_k_means(KMeans(init='random', n_clusters=n_digits, n_init=10),
              #name="random", data=data)

# in this case the seeding of the centers is deterministic, hence we run the
# kmeans algorithm only once with n_init=1
# pca = PCA(n_components=n_digits).fit(data)
# bench_k_means(KMeans(init=pca.components_, n_clusters=n_digits, n_init=1),
#               name="PCA-based",
#               data=data)
# print(79 * '_')
#
# ###############################################################################
# # Visualize the results on PCA-reduced data
#
# reduced_data = PCA(n_components=2).fit_transform(data)
# kmeans = KMeans(init='k-means++', n_clusters=n_digits, n_init=10)
# kmeans.fit(reduced_data)
#
# # Step size of the mesh. Decrease to increase the quality of the VQ.
# h = .02     # point in the mesh [x_min, m_max]x[y_min, y_max].
#
# # Plot the decision boundary. For that, we will assign a color to each
# x_min, x_max = reduced_data[:, 0].min() - 1, reduced_data[:, 0].max() + 1
# y_min, y_max = reduced_data[:, 1].min() - 1, reduced_data[:, 1].max() + 1
# xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
#
# # Obtain labels for each point in mesh. Use last trained model.
# Z = kmeans.predict(np.c_[xx.ravel(), yy.ravel()])
#
# # Put the result into a color plot
# Z = Z.reshape(xx.shape)
# plt.figure(1)
# plt.clf()
# plt.imshow(Z, interpolation='nearest',
#            extent=(xx.min(), xx.max(), yy.min(), yy.max()),
#            cmap=plt.cm.Paired,
#            aspect='auto', origin='lower')
#
# plt.plot(reduced_data[:, 0], reduced_data[:, 1], 'k.', markersize=2)
# # Plot the centroids as a white X
# centroids = kmeans.cluster_centers_
# plt.scatter(centroids[:, 0], centroids[:, 1],
#             marker='x', s=169, linewidths=3,
#             color='w', zorder=10)
# plt.title('K-means clustering on the digits dataset (PCA-reduced data)\n'
#           'Centroids are marked with white cross')
# plt.xlim(x_min, x_max)
# plt.ylim(y_min, y_max)
# plt.xticks(())
# plt.yticks(())
# plt.show()