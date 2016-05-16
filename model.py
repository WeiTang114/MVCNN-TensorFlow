#coding=utf-8
import tensorflow as tf
import re
import numpy as np


FLAGS = tf.app.flags.FLAGS
# Basic model parameters.
tf.app.flags.DEFINE_integer('batch_size', 50,
                            """Number of images to process in a batch.""")
tf.app.flags.DEFINE_float('learning_rate', 0.01,
                            """Initial learning rate.""")

# Global constants describing the CIFAR-10 data set.
IMAGE_SIZE = 100
NUM_CLASSES = 90
# NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = 
# NUM_EXAMPLES_PER_EPOCH_FOR_EVAL = cifar10_input.NUM_EXAMPLES_PER_EPOCH_FOR_EVAL

# Constants describing the training process.
MOVING_AVERAGE_DECAY = 0.9999     # The decay to use for the moving average.
NUM_EPOCHS_PER_DECAY = 350.0      # Epochs after which learning rate decays.
LEARNING_RATE_DECAY_FACTOR = 0.1  # Learning rate decay factor.

TOWER_NAME = 'tower'
DEFAULT_PADDING = 'SAME'

def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

def _activation_summary(x):
    """Helper to create summaries for activations.
    Creates a summary that provides a histogram of activations.
    Creates a summary that measure the sparsity of activations.
    Args:
      x: Tensor
    Returns:
      nothing
    """
    # Remove 'tower_[0-9]/' from the name in case this is a multi-GPU training
    # session. This helps the clarity of presentation on tensorboard.
    tensor_name = re.sub('%s_[0-9]*/' % TOWER_NAME, '', x.op.name)
    tf.histogram_summary(tensor_name + '/activations', x)
    tf.scalar_summary(tensor_name + '/sparsity', tf.nn.zero_fraction(x))

def _variable_on_cpu(name, shape, initializer):
    """Helper to create a Variable stored on CPU memory.
    Args:
      name: name of the variable
      shape: list of ints
      initializer: initializer for Variable
    Returns:
      Variable Tensor
    """
    with tf.device('/cpu:0'):
        var = tf.get_variable(name, shape, initializer=initializer)
    return var


def _variable_with_weight_decay(name, shape, stddev, wd):
    """Helper to create an initialized Variable with weight decay.
    Note that the Variable is initialized with a truncated normal distribution.
    A weight decay is added only if one is specified.
    Args:
      name: name of the variable
      shape: list of ints
      stddev: standard deviation of a truncated Gaussian
      wd: add L2Loss weight decay multiplied by this float. If None, weight
          decay is not added for this Variable.
    Returns:
      Variable Tensor
    """
    var = _variable_on_cpu(name, shape,
                           tf.truncated_normal_initializer(stddev=stddev))
    if wd:
        weight_decay = tf.mul(tf.nn.l2_loss(var), wd, name='weight_loss')
        tf.add_to_collection('losses', weight_decay)
    return var




def _conv(name, in_, ksize, strides=[1,1,1,1], padding=DEFAULT_PADDING):
    
    n_kern = ksize[3]

    with tf.variable_scope(name, reuse=False) as scope:
        stddev = 1 / np.prod(ksize[:3], dtype=float) ** 0.5
        kernel = _variable_with_weight_decay('weights', shape=ksize, stddev=stddev, wd=0.0)
        conv = tf.nn.conv2d(in_, kernel, strides, padding=padding)
        biases = _variable_on_cpu('biases', [n_kern], tf.constant_initializer(0.0))
        bias = tf.nn.bias_add(conv, biases)
        conv = tf.nn.relu(bias, name=scope.name)
        _activation_summary(conv)

    print name, conv.get_shape().as_list()
    return conv

def _maxpool(name, in_, ksize, strides, padding=DEFAULT_PADDING):
    pool = tf.nn.max_pool(in_, ksize=ksize, strides=strides,
                          padding=padding, name=name)

    print name, pool.get_shape().as_list()
    return pool

def _fc(name, in_, outsize):
    with tf.variable_scope(name, reuse=False) as scope:
        # Move everything into depth so we can perform a single matrix multiply.
        
        insize = in_.get_shape().as_list()[-1]
        stddev = 1 / float(insize) ** 0.5
        weights = _variable_with_weight_decay('weights', shape=[insize, outsize],
                                              stddev=stddev, wd=0.004)
        biases = _variable_on_cpu('biases', [outsize], tf.constant_initializer(0.0))
        fc = tf.nn.relu(tf.matmul(in_, weights) + biases, name=scope.name)
        _activation_summary(fc)

    print name, fc.get_shape().as_list()
    return fc
    


def inference_multiview(views):
    """
    views: N x V x W x H x C tensor
    """

    V = FLAGS.n_views;

    # transpose views : (NxVxWxHxC) -> (VxNxWxHxC)
    views = tf.transpose(views, perm=[1, 0, 2, 3, 4])
    
    view_pool = []
    for i in xrange(V):
        p = '_view%d' % i
        view = tf.gather(views, i) # NxWxHxC

        conv1 = _conv('conv1'+p, view, [11, 11, 3, 96], [1, 4, 4, 1], 'VALID')
        lrn1 = None
        pool1 = _maxpool('pool1'+p, conv1, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='VALID')

        conv2 = _conv('conv2'+p, pool1, [5, 5, 96, 256])
        lrn2 = None
        pool2 = _maxpool('pool2'+p, conv2, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='VALID')
        
        conv3 = _conv('conv3'+p, pool2, [3, 3, 256, 384])
        conv4 = _conv('conv4'+p, conv3, [3, 3, 384, 384])
        conv5 = _conv('conv5'+p, conv4, [3, 3, 384, 256])

        pool5 = _maxpool('pool5'+p, conv5, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='VALID')
        
        dim = 1
        for d in pool5.get_shape().as_list()[1:]:
            dim *= d

        reshape = tf.reshape(pool5, [FLAGS.batch_size, dim])
        
        view_pool.append(reshape)


    pool5_vp = _view_pool(view_pool, 'pool5_vp')
    print 'pool5_vp', pool5_vp.get_shape().as_list()


    fc6 = _fc('fc6', pool5_vp, 4096)
    fc7 = _fc('fc7', fc6, 4096)
    fc8 = _fc('fc8', fc7, 40)

    return fc8 
    

def load_alexnet_to_mvcnn(sess, caffetf_modelpath):
    """ caffemodel: np.array, """
    V = FLAGS.n_views

    def load(name, layer_data, group=1):
        w, b = layer_data

        if group != 1:
            w = np.concatenate((w, w), axis=2) 

        with tf.variable_scope(name, reuse=True):
            for subkey, data in zip(('weights', 'biases'), (w, b)):
                print 'loading ', name, subkey
                var = tf.get_variable(subkey)
                sess.run(var.assign(data))

    caffemodel = np.load(caffetf_modelpath)
    data_dict = caffemodel.item()
    for v in xrange(V):
        for l in ['conv1', 'conv2', 'conv3', 'conv4', 'conv5']:
            name = l + '_view%d' % v

            # historical grouping by alexnet
            if l == 'conv2' or l == 'conv4' or l == 'conv5':
                load(name, data_dict[l], group=2)
            else:
                load(name, data_dict[l])

    
    for l in ['fc6', 'fc7']:
        load(l, data_dict[l])
    

def _view_pool(view_features, name):
    vp = tf.expand_dims(view_features[0], 0) # eg. [100] -> [1, 100]
    for v in view_features[1:]:
        v = tf.expand_dims(v, 0)
        vp = tf.concat(0, [vp, v])
    print 'vp before reducing:', vp.get_shape().as_list()
    vp = tf.reduce_max(vp, [0], name=name)
    return vp 


def loss(fc8, labels):
    l = tf.nn.sparse_softmax_cross_entropy_with_logits(fc8, labels)
    l = tf.reduce_mean(l)
    
    tf.add_to_collection('losses', l)

    return tf.add_n(tf.get_collection('losses'), name='total_loss')


def classify(fc8):
    softmax = tf.nn.softmax(fc8)
    y = tf.argmax(softmax, 1)
    return y

def accuracy(preds, labels):
    correct_prediction = tf.equal(preds, labels)
    acc = tf.reduce_mean(tf.cast(correct_prediction, 'float'))
    return acc
    

def _add_loss_summaries(total_loss):
    """Add summaries for losses in CIFAR-10 model.
    Generates moving average for all losses and associated summaries for
    visualizing the performance of the network.
    Args:
    total_loss: Total loss from loss().
    Returns:
    loss_averages_op: op for generating moving averages of losses.
    """
    # Compute the moving average of all individual losses and the total loss.
    loss_averages = tf.train.ExponentialMovingAverage(0.9, name='avg')
    losses = tf.get_collection('losses')
    print 'losses:', losses
    loss_averages_op = loss_averages.apply(losses + [total_loss])
    # Attach a scalar summary to all individual losses and the total loss; do the
    # same for the averaged version of the losses.
    for l in losses + [total_loss]:
        # Name each loss as '(raw)' and name the moving average version of the loss
        # as the original loss name.
        tf.scalar_summary(l.op.name +' (raw)', l)
        tf.scalar_summary(l.op.name, loss_averages.average(l))
    return loss_averages_op
    

def train(total_loss, global_step, data_size):
    num_batches_per_epoch = data_size / FLAGS.batch_size
    decay_steps = int(num_batches_per_epoch * NUM_EPOCHS_PER_DECAY)

    lr = tf.train.exponential_decay(FLAGS.learning_rate,
                                    global_step,
                                    decay_steps,
                                    LEARNING_RATE_DECAY_FACTOR,
                                    staircase=True)
    tf.scalar_summary('learning_rate', lr)
    
    loss_averages_op = _add_loss_summaries(total_loss)

    with tf.control_dependencies([loss_averages_op]):
        opt = tf.train.GradientDescentOptimizer(lr)
        grads = opt.compute_gradients(total_loss)

    
    # apply gradients
    apply_gradient_op = opt.apply_gradients(grads, global_step=global_step)

    
    for var in tf.trainable_variables():
        tf.histogram_summary(var.op.name, var)

    for grad,var in grads:
        if grad:
            tf.histogram_summary(var.op.name + '/gradients', grad)

    variable_averages = tf.train.ExponentialMovingAverage(
            MOVING_AVERAGE_DECAY, global_step)

    variable_averages_op = variable_averages.apply(tf.trainable_variables())

    with tf.control_dependencies([apply_gradient_op, variable_averages_op]):
        train_op = tf.no_op(name='train')

    return train_op
