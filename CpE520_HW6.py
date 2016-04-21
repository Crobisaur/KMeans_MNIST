__author__= 'Christo Robison'

import numpy as np
import h5py
import matplotlib.pyplot as plt
import collections


f = h5py.File('mnist40_Centroids.h5','r')
centroids = f['mnist40_Centroids_numpy'][:]
f.close()
f = h5py.File('mnist_train.h5','r')
mnist_train = f['data'][:]
mnist_train_labels = f['labels'][:]
f.close()
f = h5py.File('mnist_test.h5','r')
mnist_test = f['data'][:]
mnist_test_labels = f['labels'][:]
f.close()

#print(np.shape(mnist_test))
#print(np.shape(mnist_train))
#print(np.shape(mnist_test_labels))
#print(np.shape(mnist_train_labels))
print(np.shape(centroids))

def computeRBFbetas(X, centers, labels):
    numNeurons = len(centers)
    out = []
    for i in range(numNeurons):
        cen = centers[i,:]
        diff = X - cen
        sqrdDiff = np.sum(np.power(diff,2),axis=0)
        #print(np.shape(sqrdDiff))
        distances = np.sqrt(sqrdDiff)
        sigma = np.mean(distances)
        out.append(sigma)
        print('Computing RBF beta value for centroid %i.' %(i+1))

    betas = np.divide(1,(np.multiply(2,np.power(out,2))))
    return betas

def computeRBFActivations(centers, betas, x):
    '''x = input vector'''
    diff = centers - x
    sqrdDist = np.sum(np.power(diff,2),axis=1)
    #print(np.shape(sqrdDist))
    z = np.exp(np.multiply(-betas,sqrdDist))
    return z

def trainRBFN(train_data, centroids, labels, n_classes, debug = False):
    betas = computeRBFbetas(train_data, centroids, labels)
    activations = []
    print('Calculating activations for %i examples.' %len(train_data))
    for i in range(len(train_data)):
        z = computeRBFActivations(centroids, betas, train_data[i,:])
        activations.append(z)
    addedBias = np.ones((len(train_data),len(centroids)+1))
    #add a column of ones (bias values) to the end of our maxtrix
    addedBias[:,:-1] = activations
    print('Done.')
    newLabels = convert_labels(labels, n_classes)

    theta = np.dot(np.linalg.pinv(np.dot(np.transpose(addedBias),addedBias)),
                np.dot(np.transpose(addedBias),newLabels))

    if debug:
        f = h5py.File("RBF_Thetas.h5", "w")
        f.create_dataset('labels', data=theta)
        f.close()

    out = collections.namedtuple('trainRBF', ['Theta', 'Betas', 'Activations', 'Centroids'])
    o = out(Theta=theta,Betas= betas, Activations = addedBias, Centroids = centroids)
    return o


def getThetas(activations, labels, n_classes):  #not needed after all
    out = []
    for i in range(n_classes):
        theta = np.dot(np.linalg.pinv(np.dot(np.transpose(activations), activations)),
                       np.dot(np.transpose(activations), labels))
        out.append(theta)
    return out


def convert_labels(labels,n_classes, debug = False):
    for j in range(n_classes):

        temp = labels == str(j)
        temp = temp.astype(int)
        if j > 0:
            conv_labels = np.append(conv_labels, temp)
            print(temp[:])
        else:
            conv_labels = temp
    print(np.shape(conv_labels))
    conv_labels = np.reshape(conv_labels, [len(labels), n_classes], order='F')
    if debug: print(np.shape(conv_labels))
    if debug:
        f = h5py.File("mnist_newLabels.h5", "w")
        f.create_dataset('labels', data=conv_labels)
        f.close()
    return conv_labels


def evaluateRBFN(Centroids, betas, theta, input, debug = False):
    '''input is 1 test example'''
    phis = computeRBFActivations(Centroids,betas,input)
    addedBias = np.ones(len(centroids) + 1)
    # add a column of ones (bias values) to the end of our maxtrix
    addedBias[:-1] = phis
    if debug: print(phis)
    z = np.dot(np.transpose(theta), addedBias)
    return z

def softMax(scores):
    return np.exp(scores)/np.sum(np.exp(scores),axis=0)

# take centroids use them as value for each node (one for each centroid)
# calculate beta values (standard deviation of each centroid) and weights for output nodes
# one for each class, 10 in this case.  One for each digit.

if __name__ == '__main__':
    #temp = computeRBFbetas(mnist_train, centroids, mnist_train_labels)
    #print(temp)
    #bar = computeRBFActivations(centroids,temp,mnist_train[1,:])
    #print(bar)
    foo = trainRBFN(mnist_train, centroids, mnist_train_labels, 10, True)
    print(np.shape(foo))
    result = []
    for i in range(len(mnist_test)):
        res = evaluateRBFN(foo.Centroids, foo.Betas, foo.Theta, mnist_test[i, :])
        result.append(res)
    max_IDX = np.argmax(result, axis=1)
    temp = map(str, max_IDX)
    class_test = mnist_test_labels == temp
    class_test = class_test.astype(int)
    performance = np.mean(class_test)

    print('RBFN classified MNIST with %f percent accuracy.' %performance)

    for i in range(1):
        plt.subplot(1, 1 / 1, i)
        plt.tick_params(
            axis='both',
            which='both',
            bottom='off',
            top='off',
            left='off',
            right='off',
            labelbottom='off',
            labelleft='off')
        imgp = plt.imshow(np.transpose(foo.Theta))
        plt.set_cmap('jet')
    plt.suptitle('Output layer Weights for %i node RBF Network.' %41)
    plt.show()