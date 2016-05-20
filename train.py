import numpy as np
import os,sys,inspect
import tensorflow as tf
import time
from datetime import datetime
import os
import hickle as hkl
import os.path as osp
from glob import glob

from input import Dataset


currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)
import model

TRAIN_HKL = './data/view/hkl/train.hkl'
TRAIN_LOL = './data/view/train_lists.txt'
VAL_LOL = './data/view/val_lists.txt'
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



def train(dataset_train, dataset_val, ckptfile='', caffemodel=''):
    print 'train() called'
    is_finetune = bool(ckptfile)
    V = FLAGS.n_views
    batch_size = FLAGS.batch_size

    dataset_train.shuffle()
    dataset_val.shuffle()
    data_size = dataset_train.size()
    print 'training size:', data_size


    with tf.Graph().as_default():
        startstep = 0 if not is_finetune else int(ckptfile.split('-')[-1])
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


        # must be after merge_all_summaries
        validation_loss = tf.placeholder('float32', shape=(), name='validation_loss')
        validation_summary = tf.scalar_summary('validation_loss', validation_loss)
        validation_acc = tf.placeholder('float32', shape=(), name='validation_accuracy')
        validation_acc_summary = tf.scalar_summary('validation_accuracy', validation_acc)

        saver = tf.train.Saver(tf.all_variables(), max_to_keep=1000)

        init_op = tf.initialize_all_variables()
        sess = tf.Session(config=tf.ConfigProto(log_device_placement=FLAGS.log_device_placement))
        
        if is_finetune:
            saver.restore(sess, ckptfile)
            print 'restore variables done'
        elif caffemodel:
            sess.run(init_op)
            model.load_alexnet_to_mvcnn(sess, caffemodel)
            print 'loaded pretrained caffemodel:', caffemodel
        else:
            # from scratch
            sess.run(init_op)
            print 'init_op done'

        summary_writer = tf.train.SummaryWriter(FLAGS.train_dir,
                                                graph_def=sess.graph_def) 

        step = startstep
        for epoch in xrange(100):
            print 'epoch:', epoch

            for batch_x, batch_y in dataset_train.batches(batch_size):
                if step >= FLAGS.max_steps:
                    break
                step += 1

                start_time = time.time()
                feed_dict = {view_: batch_x,
                             y_ : batch_y,
                             keep_prob_: 0.5 }

                _, pred, loss_value = sess.run(
                        [train_op, prediction,  loss,],
                        feed_dict=feed_dict)

                duration = time.time() - start_time

                assert not np.isnan(loss_value), 'Model diverged with loss = NaN'

                if step % 10 == 0:
                    sec_per_batch = float(duration)
                    print '%s: step %d, loss=%.2f (%.1f examples/sec; %.3f sec/batch)' \
                         % (datetime.now(), step, loss_value,
                                    FLAGS.batch_size/duration, sec_per_batch)

                        
                # val
                if step % 100 == 0:# and step > 0:
                    val_losses = []
                    predictions = np.array([])
                    
                    val_y = []
                    for val_step, (val_batch_x, val_batch_y) in \
                            enumerate(dataset_val.sample_batches(batch_size, VAL_SAMPLE_SIZE)):
                        val_feed_dict = {view_: val_batch_x,
                                         y_  : val_batch_y,
                                         keep_prob_: 1.0 }
                        val_loss, pred = sess.run([loss, prediction], feed_dict=val_feed_dict)
                        # print val_batch_y[:20]
                        # print pred[:20]
                        val_losses.append(val_loss)
                        predictions = np.hstack((predictions, pred))
                        val_y.extend(val_batch_y)

                    val_loss = np.mean(val_losses)
                    acc = sess.run(accuracy, feed_dict={prediction: np.array(predictions), y_:val_y[:predictions.size]})
                    print '%s: step %d, validation loss=%.2f, acc=%f' %\
                            (datetime.now(), step, val_loss, acc*100.)

                    # validation summary
                    val_loss_summ = sess.run(validation_summary,
                            feed_dict={validation_loss: val_loss})
                    val_acc_summ = sess.run(validation_acc_summary, 
                            feed_dict={validation_acc: acc})
                    summary_writer.add_summary(val_loss_summ, step)
                    summary_writer.add_summary(val_acc_summ, step)
                    summary_writer.flush()


                if step % 20 == 0:
                    # print 'running fucking summary'
                    summary_str = sess.run(summary_op, feed_dict=feed_dict)
                    summary_writer.add_summary(summary_str, step)
                    summary_writer.flush()

                if step % 200  == 0 or (step+1) == FLAGS.max_steps \
                        and step > startstep:
                    checkpoint_path = os.path.join(FLAGS.train_dir, 'model.ckpt')
                    saver.save(sess, checkpoint_path, global_step=step)



def main(argv):
    st = time.time() 
    print 'start loading data'

    listfiles_train, labels_train = read_lists(TRAIN_LOL)
    listfiles_val, labels_val = read_lists(VAL_LOL)
    dataset_train = Dataset(listfiles_train, labels_train, subtract_mean=False, V=12)
    dataset_val = Dataset(listfiles_val, labels_val, subtract_mean=False, V=12)

    print 'done loading data, time=', time.time() - st

    FLAGS.batch_size = 32

    train(dataset_train, dataset_val, FLAGS.weights, FLAGS.caffemodel)


def read_lists(list_of_lists_file):
    listfile_labels = np.loadtxt(list_of_lists_file, dtype=str).tolist()
    listfiles, labels  = zip(*[(l[0], int(l[1])) for l in listfile_labels])
    return listfiles, labels
    


if __name__ == '__main__':
    main(sys.argv)

