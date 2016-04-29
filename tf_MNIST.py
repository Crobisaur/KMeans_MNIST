__author__ = 'Christo Robison'
'''Example to practice tensorFlow for research example'''
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

import tensorflow as tf




# set globals

BATCH_SIZE = 64
NUM_CLASSES = 10
IMAGE_SIZE = 28*28

# flags = tf.app.flags
# FLAGS = flags.FLAGS
#
# flags.DEFINE_integer('max_steps', 1000, 'Number of steps to run trainer.')
# flags.DEFINE_float('learning_rate', 0.001, 'Initial learning rate.')
# flags.DEFINE_float('dropout', 0.9, 'Keep probability for training dropout.')
# flags.DEFINE_string('summaries_dir', '/home/crob/KMeans_MNIST/mnist_logs', 'Summaries directory')
#
# def variable_summaries(var, name):
#     with tf.name_scope("summaries"):
#         mean = tf.reduce_mean(var)
#         tf.scalar_summary('mean/' + name, mean)
#         with tf.name_scope('stddev'):
#             stddev = tf.sqrt(tf.reduce_sum(tf.square(var - mean)))
#         tf.scalar_summary('stddev/' + name, stddev)
#         tf.scalar_summary('max/' + name, tf.reduce_max(var))
#         tf.scalar_summary('min/' + name, tf.reduce_min(var))
#         tf.histogram_summary(name, var)
#
# def nn_layer(input_tensor, input_dim, output_dim, layer_name):
#     '''Reuseable code for making a simple NN layer
#
#     It does a matrix multiply, bias add, and then uses RELU to nonlinearize.
#     It also sets up name scoping so that the resultant graph is easy to read, and
#     adds a number of summary ops.
#     '''
#     # addig a name scpe ensures logical grouping of th elayers in the graph.
#     with tf.name_scope(layer_name):
#         # This variable will hold the state of the wights for the layer
#         with tf.name_scope("weights"):
#             weights = weight_variable([input_dim, output_dim])
#             variable_summaries(weights, layer_name + '/weights')
#         with tf.name_scope("biases"):
#             biases = bias_variable([output_dim])
#             variable_summaries(biases, layer_name + '/biases')
#         with tf.name_scope('Wx_plus_b'):
#             activations = tf.matmul(input_tensor, weights) + biases
#             tf.histogram_summary(layer_name + '/activations', activations)
#         relu = tf.nn.relu(activations, 'relu')
#         tf.histogram_summary(layer_name + '/activations_relu', relu)
#         return tf.nn.dropout(relu, keep_prob)
#
# layer1 = nn_layer(x, 784, 50, 'layer1')
# layer2 = nn_layer(layer1, 50, 10, 'layer2')
# y = tf.nn.softmax(layer2, 'predictions')
#
# with tf.name_scope('cross_entropy'):
#     diff = y_ * tf.log(y)
#     with tf.name_scope('total'):
#         cross_entropy = -tf.reduce_sum(diff)
#     with tf.name_scope('normalized'):
#         normalized_cross_entropy = -tf.reduce_mean(diff)
#     tf.scalar_summary('cross entropy', normalized_cross_entropy)
#
# with tf.name_scope('train'):
#     train_step = tf.train.AdamOptimizer(
#         FLAGS.learning_rate).minimize(cross_entropy)
#
# with tf.name_scope('accuracy'):
#     with tf.name_scope('correct_prediction'):
#         correct_predicion = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
#     with tf.name_scope('accuracy'):
#         accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
#     tf.scalar_summary('accuracy', accuracy)
#
# # Merge all the summaries and write them out to a dir
# merged = tf.mere_all_summaries()
# train_writer = tf.train.SummaryWriter('home/crob/KMeans_MNIST/mnist_logs/' + FLAGS.summaries_dir + '/train', sess.graph)
# test_writer = tf.train.SummaryWriter('home/crob/KMeans_MNIST/mnist_logs/' FLAGS.summaries_dir + '/test')
# tf.initialize_all_variables().run()
#
#





# set up config to allow for smaller memory capacity of non Tesla cards
init = tf.initialize_all_variables()

gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=.750, allocator_type='BFC', allow_growth=True)
#config = tf.ConfigProto(gpu_options)
#config.gpu_options.allocator_type='BFC'


#sess = tf.InteractiveSession(config=config)

with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
    sess.run(init)
    # start by building placeholders, datastructures friendly with
    # external GPU data loading (sort of)

    # first dimension is None b/c this list of vectors can be any length
    # however, if you don't have the VRAM you need to limit data into batches
    # x stores our mnist data as flattened image vectors 28x28 = 784

    train_data_node = tf.placeholder(tf.float32,
                                     shape=[BATCH_SIZE, IMAGE_SIZE])
    # another placeholder to hold output node labels
    train_labels_node = tf.placeholder(tf.float32,
                                       shape=[BATCH_SIZE, NUM_CLASSES])

    test_data_node = tf.placeholder(tf.float32,
                                    shape=[BATCH_SIZE, IMAGE_SIZE])
    # another placeholder to hold output node labels
    test_labels_node = tf.placeholder(tf.float32,
                                      shape=[BATCH_SIZE, NUM_CLASSES])

    validation_data_node = tf.placeholder(tf.float32,
                                          shape=[BATCH_SIZE, IMAGE_SIZE])
    # another placeholder to hold output node labels
    validation_labels_node = tf.placeholder(tf.float32,
                                            shape=[BATCH_SIZE, NUM_CLASSES])


    x = tf.placeholder(tf.float32, shape=[None, IMAGE_SIZE], name="x-input")



    # define weights matrix and bias values as variables
    # variables live inside the tensorflow graph, they aren't for
    # data storage, rather weight and parameter storage
    w = tf.Variable(tf.zeros([784, 10]), name="weights")
    b = tf.Variable(tf.zeros([10]), name="bias")

    # variables need to be initialized after a session has started.
    #sess.run(tf.initialize_all_variables())

    # use a name scope to organize nodes in the graph visualizer
    # perform softmax regression using a single line, how nice.
    with tf.name_scope("Wx_b") as scope:
        y = tf.nn.softmax(tf.matmul(x, w) + b)

    # Add summary ops to collect data
    w_hist = tf.histogram_summary("weights", w)
    b_hist = tf.histogram_summary("biases", b)
    y_hist = tf.histogram_summary("y", y)


    # another placeholder to hold output node labels
    y_ = tf.placeholder(tf.float32, shape=[None, NUM_CLASSES], name="y-input")
    # cost functions can be implemented just as easily.
    with tf.name_scope("xent") as scope:
        cross_entropy = -tf.reduce_sum(y_*tf.log(y))
        ce_sum = tf.scalar_summary("cross entropy", cross_entropy)

    # now train the model using a good ol' gradient descent
    # This line adds new operations to our graph such as update weights,
    # update steps in the epoch, etc...
    with tf.name_scope("train") as scope:
        train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)

    with tf.name_scope("test") as scope:
        correct_predicion = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
        accuracy = tf.reduce_mean(tf.cast(correct_predicion, tf.float32))
        accuracy_summary = tf.scalar_summary("accuracy", accuracy)

    # merge all the summaries and write them out to /tmp/mnist_log
    merged = tf.merge_all_summaries()
    writer = tf.train.SummaryWriter("/home/crob/KMeans_MNIST/mnist_logs", sess.graph)
    tf.initialize_all_variables().run()

    # since train_step does all of the busywork in updating weights,
    # all we have to do is call it over and over again until we reach an
    # optimal training output
    for i in range(1000):
        if i % 10 == 0: #record summary data, and the accuracy
            feed = {x: mnist.test.images, y_: mnist.test.labels}
            result = sess.run([merged, accuracy], feed_dict=feed)
            summary_str = result[0]
            acc = result[1]
            writer.add_summary(summary_str, i)
            print("Accuracy at step %s: %s" % (i, acc))
        else:
            batch_xs, batch_ys = mnist.train.next_batch(BATCH_SIZE)
            feed = {x: batch_xs, y_: batch_ys}
            sess.run(train_step, feed_dict=feed)

    writer.flush()
    writer.close()
        #batch = mnist.train.next_batch(50)
        #train_step.run(feed_dict={x: batch[0], y_: batch[1]})

    # loading 50 training examples each epoch [before weights get updated]

    # see how well our model performs.

    #accuracy = tf.reduce_mean(tf.cast(correct_predicion, tf.float32))
    print(accuracy.eval({x: mnist.test.images, y_: mnist.test.labels}))
########################################################################

    # build multilayer CNN
    # def weight_variable(shape):
    #     initial = tf.truncated_normal(shape, stddev=0.1)
    #     return tf.Variable(initial)
    #
    # def bias_variable(shape):
    #     initial = tf.constant(0.1, shape=shape)
    #     return tf.Variable(initial)
    #
    # # define our convolution operations & pooling operations
    # # padding is for when window travels along edge of image
    # # SAME needs to be all caps, kinda dumb
    # def conv2d(x, W):
    #     return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')
    #
    # def max_pool_2x2(x):
    #     return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
    #                           strides=[1, 2, 2, 1], padding='SAME')
    #
    # # build a convolutional layer
    # # 32 features with 5x5 patches
    # W_conv1 = weight_variable([5, 5, 1, 32])
    # b_conv1 = bias_variable([32])
    #
    # # reshape vector back into image so convolution can go easier
    # x_image = tf.reshape(x,[-1, 28, 28, 1])
    #
    # # set up our convolution operation as well as pooling layer
    # h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
    # h_pool1 = max_pool_2x2(h_conv1)
    #
    # # build second cnn layer
    #
    # W_conv2 = weight_variable([5, 5, 32, 64])
    # b_conv2 = bias_variable([64])
    #
    # h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
    # h_pool2 = max_pool_2x2(h_conv2)
    #
    # # now we build a densely (fully) connected layer to make sense of our
    # # convolutions
    #
    # w_fc1 = weight_variable([7 * 7 * 64, 1024])
    # b_fc1 = bias_variable([1024])
    #
    # h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
    # h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, w_fc1) + b_fc1)
    #
    # # now we can implement drop out, super neat since I haven't done it myself yet
    # keep_prob = tf.placeholder(tf.float32)
    # h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)
    #
    # # once again, we send our output to a final softmax layer to get an
    # # output we can interperet
    #
    # W_fc2 = weight_variable([1024, NUM_CLASSES])
    # b_fc2 = bias_variable([NUM_CLASSES])
    #
    # y_conv = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)
    #
    # # finally run a loop to train & validate our CNN
    # cross_entropy = -tf.reduce_sum(y_*tf.log(y_conv))
    # train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
    # correct_predicion = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
    # accuracy = tf.reduce_mean(tf.cast(correct_predicion, tf.float32))
    # sess.run(tf.initialize_all_variables())
    #
    # for i in range(2000):
    #     batch = mnist.train.next_batch(BATCH_SIZE)
    #     if i%100 == 0:
    #         train_accuracy = accuracy.eval(feed_dict={
    #             x:batch[0], y_:batch[1], keep_prob: 1.0})
    #         print("step %d, training accuracy %g"%(i, train_accuracy))
    #     train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})
    #
    # print("test accuracy %g"%accuracy.eval(feed_dict={
    #     x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0}))