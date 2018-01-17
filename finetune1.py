#written by pengkai.wang
import os
import math
from datetime import datetime
import numpy as np
import tensorflow as tf
from inception_resnet_v2 import inception_resnet_v2
from inception_resnet_v2 import inception_resnet_v2_arg_scope
import inception_preprocessing

os.environ['CUDA_VISIBLE_DEVICES']='1'

image_size=inception_resnet_v2.default_image_size

tf.app.flags.DEFINE_integer('num_classes', 14, 'the number of classes')
tf.app.flags.DEFINE_string('train_tfrecords' , '/data0/users/pengkai1/datasets/MultiLabel/train.tfrecords','train_tfrecords')
tf.app.flags.DEFINE_float('learning_rate', 0.00001, 'the learning rate of the network')
tf.app.flags.DEFINE_integer('batch_size',32, 'the batch size during training the network')
tf.app.flags.DEFINE_integer('epoch', 60, 'the epoch of training')
tf.app.flags.DEFINE_string('checkpoint', 'inception_resnet_v2_2016_08_30.ckpt', 'checkpoint file')
tf.app.flags.DEFINE_string('finetune_output_model_dir', 'finetune_output_model_dir', 'finetune out model dir')
tf.app.flags.DEFINE_float('threshold', 0.5, 'threshold for training and testing')

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
    #image_data = tf.cast(image_data, tf.float32)
    image_data = tf.reshape(image_data, image_shape)

    image_label=tf.decode_raw(feature['image_label'],tf.float32)
    image_label=tf.reshape(image_label,[FLAGS.num_classes])

    #print image_data
    #print image_label
    return image_data,image_label
    pass

#config=tf.ConfigProto()
#config.gpu_options.per_process_gpu_memory_fraction=0.3
def get_variables_to_restore():
    exclude_scopes = ['InceptionResnetV2/AuxLogits', 'InceptionResnetV2/Logits']
    variables_to_restore=[]
    for var in tf.trainable_variables():
        excluded=False
        for exclude_scope in exclude_scopes:
            if var.op.name.startswith(exclude_scope):
                excluded=True
                break
        if not excluded:
            variables_to_restore.append(var)
    return variables_to_restore

def main(unused_argv):
    with tf.Graph().as_default():
        #tf.logging.set_verbosity(tf.logging.DEBUG)

        image, label = parse_single_image(FLAGS.train_tfrecords)
        #image.set_shape([None, None, 3])
        #image = tf.image.resize_images(image, [image_size, image_size])
        # Preprocess images and scaling images
        image = inception_preprocessing.preprocess_image(image, image_size, image_size,
                                                         is_training=True)

        #images, labels = tf.train.shuffle_batch([image, label], batch_size=FLAGS.batch_size, num_threads=2,
                                               # capacity=50000, min_after_dequeue=10000)
        images, labels = tf.train.batch([image, label], batch_size=FLAGS.batch_size, num_threads=2,
                                                capacity=10000)

        #input_images=tf.placeholder(tf.float32,shape=[FLAGS.batch_size,image_size,image_size,3])
        #targets=tf.placeholder(tf.float32,shape=[FLAGS.batch_size,FLAGS.num_classes])
        # create the model
        with tf.contrib.framework.arg_scope(inception_resnet_v2_arg_scope()):
            logits, _ = inception_resnet_v2(images, num_classes=FLAGS.num_classes, is_training=True)
        predictions_score = tf.nn.sigmoid(logits, name='predictions')
        cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(labels=labels, logits=logits)
        loss = tf.reduce_mean(cross_entropy)

        global_steps = tf.Variable(0, name='global_steps', trainable=False)

        trainable_scopes = ['InceptionResnetV2/AuxLogits', 'InceptionResnetV2/Logits']
        variables_to_train = []
        for trainable_scope in trainable_scopes:
            variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, trainable_scope)
            variables_to_train.extend(variables)

        #optimizer = tf.train.GradientDescentOptimizer(learning_rate=FLAGS.learning_rate, name='GradientDescent')
        optimizer = tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate, name='AdamOptimizer')
        train_op=optimizer.minimize(loss,global_step=global_steps,var_list=variables_to_train)

        prediction_labels=tf.greater_equal(predictions_score,FLAGS.threshold)
        correct_prediction=tf.equal(prediction_labels,tf.equal(labels,1.0))
        correct_prediction=tf.cast(correct_prediction,tf.float32)
        total_accuracy=tf.reduce_mean(correct_prediction)

        saver=tf.train.Saver(get_variables_to_restore())
        finetune_model_saver=tf.train.Saver()
        init=tf.initialize_all_variables()

        with tf.Session() as sess:
            sess.run(init)
            #restore the inception_resnet_v2 model excepting the last layer
            saver.restore(sess, FLAGS.checkpoint)
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(sess=sess, coord=coord)
            train_steps_per_epoch = math.ceil(21510 / FLAGS.batch_size)
            #train_steps_per_epoch = math.ceil(10 / FLAGS.batch_size)
            number_of_train_steps = FLAGS.epoch * int(train_steps_per_epoch)

            for i in range(number_of_train_steps):
                #images_batch = sess.run(images)
                #labels_batch = sess.run(labels)
                current_step = sess.run(global_steps)
                _,train_logits,train_predictions_score,train_prediction_labels, train_labels, train_loss, total_train_accuracy = sess.run([train_op, logits, predictions_score,prediction_labels, labels, loss, total_accuracy])
                #print("train_labels=%s" % train_labels)
                print('train_logits=%s' % train_logits)
                print('train_predictions_score=%s' % train_predictions_score)
                print('train_prediction_labels=%s' % train_prediction_labels)
                print('train_labels=%s' % train_labels)
                print('%s step=%d, train_loss=%f, total_train_accuracy=%.5f' % (datetime.now(),current_step+1, train_loss, total_train_accuracy * 100))

            finetune_model_saver.save(sess,os.path.join(FLAGS.finetune_output_model_dir,'finetune_model_'+str(current_step+1)+'.ckpt'))
            coord.request_stop()
            coord.join(threads)
            print('Finetune the model Done!')

if __name__=='__main__':
    tf.app.run()