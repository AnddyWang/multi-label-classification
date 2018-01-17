#written by pengkai.wang
import os
import math
from datetime import datetime
import numpy as np
import tensorflow as tf
from inception_resnet_v2 import inception_resnet_v2
from inception_resnet_v2 import inception_resnet_v2_arg_scope

os.environ['CUDA_VISIBLE_DEVICES']='1'

image_size=inception_resnet_v2.default_image_size

tf.app.flags.DEFINE_integer('num_classes',14,'the number of classes')
tf.app.flags.DEFINE_string('val_tfrecords','/data0/users/pengkai1/datasets/MultiLabel/test.tfrecords','test_tfrecords')
tf.app.flags.DEFINE_float('learning_rate',0.001,'the learning rate of the network')
tf.app.flags.DEFINE_integer('batch_size',20,'the batch size during training the network')
tf.app.flags.DEFINE_integer('epoch',60,'the epoch of training')
tf.app.flags.DEFINE_string('checkpoint','inception_resnet_v2_2016_08_30.ckpt','checkpoint file')
tf.app.flags.DEFINE_string('output_model_dir','output_model_dir','out model dir')
tf.app.flags.DEFINE_float('threshold',0.5,'threshold for training and testing')

FLAGS=tf.app.flags.FLAGS

def main(unused_argv):
    with tf.Graph().as_default():
        #tf.logging.set_verbosity(tf.logging.DEBUG)

        image.set_shape([None, None, 3])
        image = tf.image.resize_images(image, [image_size, image_size])

        input_images=tf.placeholder(tf.float32,shape=[FLAGS.batch_size,image_size,image_size,3])
        targets=tf.placeholder(tf.float32,shape=[FLAGS.batch_size,FLAGS.num_classes])
        # create the model
        with tf.contrib.framework.arg_scope(inception_resnet_v2_arg_scope()):
            logits, _ = inception_resnet_v2(input_images, num_classes=FLAGS.num_classes, is_training=False)
        predictions = tf.nn.sigmoid(logits, name='predictions')
        cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(labels=targets, logits=logits)
        loss = tf.reduce_mean(cross_entropy)

        saver=tf.train.Saver()
        with tf.Session() as sess:
            #restore the inception_resnet_v2 model excepting the last layer
            saver.restore(sess, os.path.join(FLAGS.output_model_dir,'finetune_model_64500.ckpt'))
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(sess=sess, coord=coord)
            val_steps_per_epoch = int(math.ceil(5377 / FLAGS.batch_size))
            val_accuracy=0
            val_count=0
            val_avg_loss=0
            for i in range(val_steps_per_epoch):
                val_loss, total_val_accuracy = sess.run([loss, total_accuracy],
                                                   feed_dict={input_images: images_batch,
                                                              targets: labels_batch})
                val_accuracy+=total_val_accuracy
                val_count+=1
                val_avg_loss+=val_loss
                print('%s step=%d, val_loss=%f, total_val_accuracy=%.5f' % (datetime.now(),i, val_loss, total_val_accuracy * 100))

            print('val_avg_loss=%.3f, total_val_accuracy=%.3f' % (val_avg_loss/val_count, (val_accuracy/val_count) * 100))
            coord.request_stop()
            coord.join(threads)

if __name__=='__main__':
    tf.app.run()