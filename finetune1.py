#written by pengkai.wang
import os
import math
import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim
from inception_resnet_v2 import inception_resnet_v2
from inception_resnet_v2 import inception_resnet_v2_arg_scope

os.environ['CUDA_VISIBLE_DEVICES']='1'

image_size=inception_resnet_v2.default_image_size

tf.app.flags.DEFINE_integer('num_classes',14,'the number of classes')
tf.app.flags.DEFINE_string('train_tfrecords','/data0/users/pengkai1/datasets/MultiLabel/train.tfrecords','train_tfrecords')
tf.app.flags.DEFINE_string('test_tfrecords','/data0/users/pengkai1/datasets/MultiLabel/test.tfrecords','test_tfrecords')
tf.app.flags.DEFINE_float('learning_rate',0.001,'the learning rate of the network')
tf.app.flags.DEFINE_integer('batch_size',20,'the batch size during training the network')
tf.app.flags.DEFINE_integer('epoch',60,'the epoch of training')
tf.app.flags.DEFINE_string('checkpoint','inception_resnet_v2_2016_08_30.ckpt','checkpoint file')
tf.app.flags.DEFINE_string('output_model_dir','output_model_dir','out model dir')

FLAGS=tf.app.flags.FLAGS

def parse_single_image(train_tfrecords):
    file_queue=tf.train.string_input_producer([train_tfrecords])
    reader=tf.TFRecordReader()
    filename,serialized_example=reader.read(file_queue)
    #print('filename=%s,serialized_example=%s',filename,serialized_example)
    feature=tf.parse_single_example(serialized_example,features={
        'image_data':tf.FixedLenFeature([],tf.string),
        'image_height':tf.FixedLenFeature([],tf.int64),
        'image_width': tf.FixedLenFeature([], tf.int64),
        'image_channel': tf.FixedLenFeature([], tf.int64),
        'image_label':tf.FixedLenFeature([],tf.string)
    })
    #print feature['image_data']
    #print feature['image_label']
    image_data=tf.decode_raw(feature['image_data'],tf.uint8)

    image_height=tf.cast(feature['image_height'],tf.int32)
    image_width=tf.cast(feature['image_width'],tf.int32)
    image_channel=tf.cast(feature['image_channel'],tf.int32)
    #print(image_height,image_width,image_channel)

    image_shape=tf.stack([image_height,image_width,image_channel])
    image_data = tf.cast(image_data, tf.float32)
    image_data = tf.reshape(image_data, image_shape)

    image_label=tf.decode_raw(feature['image_label'],tf.float32)
    image_label=tf.reshape(image_label,[FLAGS.num_classes])

    #print image_data
    #print image_label
    return image_data,image_label
    pass

#config=tf.ConfigProto()
#config.gpu_options.per_process_gpu_memory_fraction=0.3
def get_init_fn(checkpoint):
    exclude_scopes = ['InceptionResnetV2/AuxLogits', 'InceptionResnetV2/Logits']
    variables_to_restore=[]
    for var in slim.get_model_variables():
        excluded=False
        for exclude_scope in exclude_scopes:
            if var.op.name.startswith(exclude_scope):
                excluded=True
                break
        if not excluded:
            variables_to_restore.append(var)
    return slim.assign_from_checkpoint_fn(checkpoint, variables_to_restore)

def main(unused_argv):
    with tf.Graph().as_default():
        tf.logging.set_verbosity(tf.logging.DEBUG)

        image, label = parse_single_image(FLAGS.train_tfrecords)
        image.set_shape([None, None, 3])
        image = tf.image.resize_images(image, [image_size, image_size])
        images, labels = tf.train.shuffle_batch([image, label], batch_size=FLAGS.batch_size, num_threads=2,
                                                capacity=50000, min_after_dequeue=10000)
        print(images)
        print(labels)
        with tf.Session() as sess:
            coord = tf.train.Coordinator()
            print('Session')
            threads = tf.train.start_queue_runners(sess=sess, coord=coord)
            coord.request_stop()
            coord.join(threads)

        # create the model
        with slim.arg_scope(inception_resnet_v2_arg_scope()):
            logits, _ = inception_resnet_v2(images, num_classes=FLAGS.num_classes, is_training=True)
        predictions = tf.nn.sigmoid(logits, name='predictions')
        cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(labels=labels, logits=logits)
        loss = tf.reduce_mean(cross_entropy)

        #print(global_step)

        optimizer = tf.train.GradientDescentOptimizer(learning_rate=FLAGS.learning_rate, name='GradientDescent')
        global_steps = tf.Variable(0, name='global_steps', trainable=False)

        trainable_scopes = ['InceptionResnetV2/AuxLogits', 'InceptionResnetV2/Logits']
        variables_to_train = []
        for trainable_scope in trainable_scopes:
            variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, trainable_scope)
            variables_to_train.extend(variables)


        train_op = slim.learning.create_train_op(loss, optimizer, variables_to_train=variables_to_train, global_step=global_steps)

        #variables_to_restore = slim.get_variables_to_restore(exclude=trainable_scopes)
        #init_fn = slim.assign_from_checkpoint_fn(FLAGS.checkpoint, variables_to_restore)

        steps_per_epoch = math.ceil(26887 / FLAGS.batch_size)
        number_of_steps = FLAGS.epoch * int(steps_per_epoch)

        slim.learning.train(train_op, FLAGS.output_model_dir, init_fn=get_init_fn(FLAGS.checkpoint), number_of_steps=number_of_steps, global_step=global_steps)

        #print('loss=%f', loss.eval(session=sess))
        #sess.close()


'''
    with tf.Session(config=config) as sess:
        coord=tf.train.Coordinator()
        tf.train.start_queue_runners(sess=sess,coord=coord)
        image = sess.run(image)
        label = sess.run(label)
        print(image)
        print(image.shape)
        print(label)
    pass'''

if __name__=='__main__':
    tf.app.run()