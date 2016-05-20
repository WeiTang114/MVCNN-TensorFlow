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

from input import Dataset


currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)
import model

TRAIN_HKL = './data/view/hkl/train.hkl'
TEST_LOL = './data/view/test_lists.txt'
VAL_SAMPLE_SIZE = 256 

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('train_dir', '/tmp3/weitang114/MVCNN-TF/tmp/',
                           """Directory where to write event logs """
                           """and checkpoint.""")
tf.app.flags.DEFINE_integer('max_steps', 1000000,
                            """Number of batches to run.""")
tf.app.flags.DEFINE_boolean('log_device_placement', False,
                            """Whether to log device placement.""")
tf.app.flags.DEFINE_string('weights', '', 
                            """finetune with a pretrained model""")
tf.app.flags.DEFINE_string('n_views', 12, 
                            """Number of views rendered from a mesh.""")
tf.app.flags.DEFINE_string('caffemodel', '', 
                            """finetune with a model converted by caffe-tensorflow""")

# Constants describing the training process.
MOVING_AVERAGE_DECAY = 0.9999     # The decay to use for the moving average.
NUM_EPOCHS_PER_DECAY = 20.0      # Epochs after which learning rate decays.
LEARNING_RATE_DECAY_FACTOR = 0.05  # Learning rate decay factor.

np.set_printoptions(precision=3)



def train(dataset, ckptfile):
    print 'train() called'
    V = FLAGS.n_views
    batch_size = FLAGS.batch_size

    data_size = dataset.size()
    print 'training size:', data_size


    with tf.Graph().as_default():
        startstep = 0
        global_step = tf.Variable(startstep, trainable=False)
         
        
        view_ = tf.placeholder('float32', shape=(None, V, 227, 227, 3), name='im0')
        y_ = tf.placeholder('int64', shape=(None), name='y')
        keep_prob_ = tf.placeholder('float32')

        fc8 = model.inference_multiview(view_, keep_prob_)
        loss = model.loss(fc8, y_)
        train_op = model.train(loss, global_step, data_size)
        prediction = model.classify(fc8)
        accuracy = model.accuracy(prediction, y_)

        # build the summary operation based on the F colection of Summaries
        summary_op = tf.merge_all_summaries()

        saver = tf.train.Saver(tf.all_variables(), max_to_keep=1000)

        init_op = tf.initialize_all_variables()
        sess = tf.Session(config=tf.ConfigProto(log_device_placement=FLAGS.log_device_placement))
        
        saver.restore(sess, ckptfile)
        print 'restore variables done'

        summary_writer = tf.train.SummaryWriter(FLAGS.train_dir,
                                                graph_def=sess.graph_def) 

        step = startstep
            
        predictions = []
        labels = []

        for batch_x, batch_y in dataset.batches(batch_size):
            if step >= FLAGS.max_steps:
                break
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

        print labels
        print predictions
        acc = metrics.accuracy_score(labels, predictions)
        print 'acc:', acc*100



def main(argv):
    st = time.time() 
    print 'start loading data'

    listfiles, labels = read_lists(TEST_LOL)
    dataset = Dataset(listfiles, labels, subtract_mean=False, V=12)

    print 'done loading data, time=', time.time() - st

    FLAGS.batch_size = 32

    train(dataset, FLAGS.weights)


def read_lists(list_of_lists_file):
    listfile_labels = np.loadtxt(list_of_lists_file, dtype=str).tolist()
    listfiles, labels  = zip(*[(l[0], int(l[1])) for l in listfile_labels])
    return listfiles, labels
    


if __name__ == '__main__':
    main(sys.argv)


