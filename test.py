import numpy as np
import os,sys,inspect
import tensorflow as tf
import time
from datetime import datetime
import os
import hickle as hkl
import os.path as osp
from glob import glob
import sklearn.metrics as metrics
import math

from input import Dataset
import globals as g_


currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)
import model

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('train_dir', osp.dirname(sys.argv[0]) + '/tmp/',
                           """Directory where to write event logs """
                           """and checkpoint.""")
tf.app.flags.DEFINE_boolean('log_device_placement', False,
                            """Whether to log device placement.""")
tf.app.flags.DEFINE_string('weights', '',
                            """finetune with a pretrained model""")

np.set_printoptions(precision=3)


def test(dataset, ckptfile):
    print 'test() called'
    V = g_.NUM_VIEWS
    batch_size = FLAGS.batch_size

    data_size = dataset.size()
    print 'dataset size:', data_size

    with tf.Graph().as_default():
        startstep = 0
        global_step = tf.Variable(startstep, trainable=False)

        view_ = tf.placeholder('float32', shape=(None, V, 227, 227, 3), name='im0')
        y_ = tf.placeholder('int64', shape=(None), name='y')
        keep_prob_ = tf.placeholder('float32')

        fc8 = model.inference_multiview(view_, g_.NUM_CLASSES, keep_prob_)
        loss = model.loss(fc8, y_)
        train_op = model.train(loss, global_step, data_size)
        prediction = model.classify(fc8)

        saver = tf.train.Saver(tf.all_variables(), max_to_keep=1000)

        init_op = tf.global_variables_initializer()
        sess = tf.Session(config=tf.ConfigProto(log_device_placement=FLAGS.log_device_placement))

        saver.restore(sess, ckptfile)
        print 'restore variables done'

        step = startstep

        predictions = []
        labels = []

        print "Start testing"
        print "Size:", data_size
        print "It'll take", int(math.ceil(data_size/batch_size)), "iterations."

        for batch_x, batch_y in dataset.batches(batch_size):
            step += 1

            start_time = time.time()
            feed_dict = {view_: batch_x,
                         y_ : batch_y,
                         keep_prob_: 1.0}

            pred, loss_value = sess.run(
                    [prediction,  loss,],
                    feed_dict=feed_dict)


            duration = time.time() - start_time

            assert not np.isnan(loss_value), 'Model diverged with loss = NaN'

            if step % 10 == 0:
                sec_per_batch = float(duration)
                print '%s: step %d, loss=%.2f (%.1f examples/sec; %.3f sec/batch)' \
                     % (datetime.now(), step, loss_value,
                                FLAGS.batch_size/duration, sec_per_batch)

            predictions.extend(pred.tolist())
            labels.extend(batch_y.tolist())

        # print labels
        # print predictions
        acc = metrics.accuracy_score(labels, predictions)
        print 'acc:', acc*100


def main(argv):
    st = time.time()
    print 'start loading data'

    listfiles, labels = read_lists(g_.TEST_LOL)
    dataset = Dataset(listfiles, labels, subtract_mean=False, V=g_.NUM_VIEWS)

    print 'done loading data, time=', time.time() - st

    test(dataset, FLAGS.weights)


def read_lists(list_of_lists_file):
    listfile_labels = np.loadtxt(list_of_lists_file, dtype=str).tolist()
    listfiles, labels  = zip(*[(l[0], int(l[1])) for l in listfile_labels])
    return listfiles, labels


if __name__ == '__main__':
    main(sys.argv)


