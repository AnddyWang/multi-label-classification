#written by pengkai.wang
import os
import numpy as np
import tensorflow as tf

os.environ['CUDA_VISIBLE_DEVICES']='1'

tf.app.flags.DEFINE_integer('num_classes',14,'the number of classes')
tf.app.flags.DEFINE_string('train_tfrecords','/data0/users/pengkai1/datasets/MultiLabel/train.tfrecords','train_tfrecords')
tf.app.flags.DEFINE_string('test_tfrecords','/data0/users/pengkai1/datasets/MultiLabel/test.tfrecords','test_tfrecords')
tf.app.flags.DEFINE_float('learning_rate',0.001,'the learning rate of the network')
tf.app.flags.DEFINE_integer('batch_size',64,'the batch size during training the network')
tf.app.flags.DEFINE_integer('epoch',60,'the epoch of training')
tf.app.flags.DEFINE_string('checkpoint','inception_resnet_v2_2016_08_30.ckpt','checkpoint file')

FLAGS=tf.app.flags.FLAGS

def parse_single_image(train_tfrecords):
    file_queue=tf.train.string_input_producer([train_tfrecords])
    reader=tf.TFRecordReader()
    filename,serialized_example=reader.read(file_queue)
    print('filename=%s,serialized_example=%s',filename,serialized_example)
    feature=tf.parse_single_example(serialized_example,features={
        'image_data':tf.FixedLenFeature([],tf.string),
        'image_height':tf.FixedLenFeature([],tf.int64),
        'image_width': tf.FixedLenFeature([], tf.int64),
        'image_channel': tf.FixedLenFeature([], tf.int64),
        'image_label':tf.FixedLenFeature([],tf.string)
    })
    print feature['image_data']
    print feature['image_label']
    image_data=tf.decode_raw(feature['image_data'],tf.uint8)

    image_height=tf.cast(feature['image_height'],tf.int32)
    image_width=tf.cast(feature['image_width'],tf.int32)
    image_channel=tf.cast(feature['image_channel'],tf.int32)
    print(image_height,image_width,image_channel)

    image_shape=tf.stack([image_height,image_width,image_channel])
    image_data = tf.cast(image_data, tf.float32)
    image_data = tf.reshape(image_data, image_shape)

    image_label=tf.decode_raw(feature['image_label'],tf.float32)
    image_label=tf.reshape(image_label,[FLAGS.num_classes])

    print image_data
    print image_label
    return image_data,image_label
    pass

config=tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction=0.3

def main(unused_argv):
    image,label = parse_single_image(FLAGS.train_tfrecords)
    images,labels=tf.train.shuffle_batch([image,label],batch_size=FLAGS.batch_size,num_threads=4,capacity=50000,min_after_dequeue=10000)
    with tf.Session(config=config) as sess:
        tf.train.start_queue_runners()
        image = sess.run(image)
        label = sess.run(label)
        print(image)
        print(image.shape)
        print(label)
    pass

if __name__=='__main__':
    tf.app.run()