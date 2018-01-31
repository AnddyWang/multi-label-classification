#written by pengkai.wang
import os
import math
from datetime import datetime
import numpy as np
import tensorflow as tf
from inception_resnet_v2 import inception_resnet_v2
from inception_resnet_v2 import inception_resnet_v2_arg_scope
import inception_preprocessing
from PIL import Image

os.environ['CUDA_VISIBLE_DEVICES']='0'

image_size=inception_resnet_v2.default_image_size

tf.app.flags.DEFINE_integer('num_classes',14,'the number of classes')
tf.app.flags.DEFINE_string('val_tfrecords','/data0/users/pengkai1/datasets/MultiLabel/test.tfrecords','test_tfrecords')
tf.app.flags.DEFINE_string('image_path','/data0/users/pengkai1/datasets/MultiLabel/images/1.jpg','the epoch of training')
tf.app.flags.DEFINE_string('checkpoint','finetune_model_2150.ckpt','checkpoint file')
#tf.app.flags.DEFINE_string('checkpoint','finetune_model_64500.ckpt','checkpoint file')
tf.app.flags.DEFINE_string('output_model_dir','output_model_dir','out model dir')
tf.app.flags.DEFINE_string('labels','labels.txt','labels.txt')

FLAGS=tf.app.flags.FLAGS

def main(unused_argv):
    with tf.Graph().as_default():
        #tf.logging.set_verbosity(tf.logging.DEBUG)
        labels=[label.strip() for label in open(FLAGS.labels,'r').readlines()]

        # Preprocess images
        image = Image.open(FLAGS.image_path, 'r')
        image = inception_preprocessing.preprocess_image(image, image_size, image_size,
                                                         is_training=False)

        # create the model
        with tf.contrib.framework.arg_scope(inception_resnet_v2_arg_scope()):
            logits, _ = inception_resnet_v2(image, num_classes=FLAGS.num_classes, is_training=False)
        predictions = tf.nn.sigmoid(logits, name='predictions')

        saver=tf.train.Saver()
        with tf.Session() as sess:
            #restore the inception_resnet_v2 model excepting the last layer
            saver.restore(sess, os.path.join(FLAGS.finetune_output_model_dir, FLAGS.checkpoint))
            predictions=sess.run(predictions)
            sort_label=predictions[0].argsort()[::-1]

            for index in sort_label:
                label_str=labels[index]
                label_score=predictions[index]
                print('label=%s,score=%.5f' % (label_str,label_score))

if __name__=='__main__':
    tf.app.run()