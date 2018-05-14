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
tf.app.flags.DEFINE_string('caffemodel', '', 
                            """finetune with a model converted by caffe-tensorflow""")

np.set_printoptions(precision=3)


def train(dataset_train, dataset_val, ckptfile='', caffemodel=''):
    print 'train() called'
    is_finetune = bool(ckptfile)
    V = g_.NUM_VIEWS
    batch_size = FLAGS.batch_size

    dataset_train.shuffle()
    dataset_val.shuffle()
    data_size = dataset_train.size()
    print 'training size:', data_size


    with tf.Graph().as_default():
        startstep = 0 if not is_finetune else int(ckptfile.split('-')[-1])
        global_step = tf.Variable(startstep, trainable=False)
         
        # placeholders for graph input
        view_ = tf.placeholder('float32', shape=(None, V, 227, 227, 3), name='im0')
        y_ = tf.placeholder('int64', shape=(None), name='y')
        keep_prob_ = tf.placeholder('float32')

        # graph outputs
        fc8 = model.inference_multiview(view_, g_.NUM_CLASSES, keep_prob_)
        loss = model.loss(fc8, y_)
        train_op = model.train(loss, global_step, data_size)
        prediction = model.classify(fc8)

        # build the summary operation based on the F colection of Summaries
        summary_op = tf.summary.merge_all()


        # must be after merge_all_summaries
        validation_loss = tf.placeholder('float32', shape=(), name='validation_loss')
        validation_summary = tf.summary.scalar('validation_loss', validation_loss)
        validation_acc = tf.placeholder('float32', shape=(), name='validation_accuracy')
        validation_acc_summary = tf.summary.scalar('validation_accuracy', validation_acc)

        saver = tf.train.Saver(tf.all_variables(), max_to_keep=1000)

        init_op = tf.global_variables_initializer()
        sess = tf.Session(config=tf.ConfigProto(log_device_placement=FLAGS.log_device_placement, gpu_options=tf.GPUOptions(allow_growth=True)))
        
        if is_finetune:
            # load checkpoint file
            saver.restore(sess, ckptfile)
            print 'restore variables done'
        elif caffemodel:
            # load caffemodel generated with caffe-tensorflow
            sess.run(init_op)
            model.load_alexnet_to_mvcnn(sess, caffemodel)
            print 'loaded pretrained caffemodel:', caffemodel
        else:
            # from scratch
            sess.run(init_op)
            print 'init_op done'

        summary_writer = tf.summary.FileWriter(FLAGS.train_dir,
                                               graph=sess.graph) 

        step = startstep
        for epoch in xrange(100):
            print 'epoch:', epoch

            for batch_x, batch_y in dataset_train.batches(batch_size):
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

                # print training information
                if step % 10 == 0 or step - startstep <= 30:
                    sec_per_batch = float(duration)
                    print '%s: step %d, loss=%.2f (%.1f examples/sec; %.3f sec/batch)' \
                         % (datetime.now(), step, loss_value,
                                    FLAGS.batch_size/duration, sec_per_batch)

                        
                # validation
                if step % g_.VAL_PERIOD == 0:# and step > 0:
                    val_losses = []
                    predictions = np.array([])
                    
                    val_y = []
                    for val_step, (val_batch_x, val_batch_y) in \
                            enumerate(dataset_val.sample_batches(batch_size, g_.VAL_SAMPLE_SIZE)):
                        val_feed_dict = {view_: val_batch_x,
                                         y_  : val_batch_y,
                                         keep_prob_: 1.0 }
                        val_loss, pred = sess.run([loss, prediction], feed_dict=val_feed_dict)
                        val_losses.append(val_loss)
                        predictions = np.hstack((predictions, pred))
                        val_y.extend(val_batch_y)

                    val_loss = np.mean(val_losses)
                    
                    acc = metrics.accuracy_score(val_y[:predictions.size], np.array(predictions))
                    print '%s: step %d, validation loss=%.4f, acc=%f' %\
                            (datetime.now(), step, val_loss, acc*100.)

                    # validation summary
                    val_loss_summ = sess.run(validation_summary,
                            feed_dict={validation_loss: val_loss})
                    val_acc_summ = sess.run(validation_acc_summary, 
                            feed_dict={validation_acc: acc})
                    summary_writer.add_summary(val_loss_summ, step)
                    summary_writer.add_summary(val_acc_summ, step)
                    summary_writer.flush()


                if step % 100 == 0:
                    # print 'running summary'
                    summary_str = sess.run(summary_op, feed_dict=feed_dict)
                    summary_writer.add_summary(summary_str, step)
                    summary_writer.flush()

                if step % g_.SAVE_PERIOD == 0 and step > startstep:
                    checkpoint_path = os.path.join(FLAGS.train_dir, 'model.ckpt')
                    saver.save(sess, checkpoint_path, global_step=step)



def main(argv):
    st = time.time() 
    print 'start loading data'

    listfiles_train, labels_train = read_lists(g_.TRAIN_LOL)
    listfiles_val, labels_val = read_lists(g_.VAL_LOL)
    dataset_train = Dataset(listfiles_train, labels_train, subtract_mean=False, V=g_.NUM_VIEWS)
    dataset_val = Dataset(listfiles_val, labels_val, subtract_mean=False, V=g_.NUM_VIEWS)

    print 'done loading data, time=', time.time() - st

    train(dataset_train, dataset_val, FLAGS.weights, FLAGS.caffemodel)


def read_lists(list_of_lists_file):
    listfile_labels = np.loadtxt(list_of_lists_file, dtype=str).tolist()
    listfiles, labels  = zip(*[(l[0], int(l[1])) for l in listfile_labels])
    return listfiles, labels
    


if __name__ == '__main__':
    main(sys.argv)

