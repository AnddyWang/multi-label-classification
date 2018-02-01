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

tf.app.flags.DEFINE_integer('num_classes',21,'the number of classes')
tf.app.flags.DEFINE_string('val_tfrecords','/data0/users/pengkai1/datasets/MultiLabelImage/train.tfrecords','test_tfrecords')
tf.app.flags.DEFINE_integer('batch_size', 4, 'the batch size during training the network')
#tf.app.flags.DEFINE_string('checkpoint','finetune_model_2150.ckpt','checkpoint file')
tf.app.flags.DEFINE_string('checkpoint','0201_finetune_model_28140.ckpt','checkpoint file')
tf.app.flags.DEFINE_string('finetune_output_model_dir','finetune_output_model_dir','out model dir')
tf.app.flags.DEFINE_float('threshold',0.5,'threshold for training and testing')
tf.app.flags.DEFINE_string('multi_label_log_dir','multi_label_log_dir','multi_label_log_dir')

FLAGS=tf.app.flags.FLAGS

def parse_single_image(tfrecords_file):
    file_queue=tf.train.string_input_producer([tfrecords_file])
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
    image_data = tf.decode_raw(feature['image_data'],tf.uint8)

    image_height = tf.cast(feature['image_height'],tf.int32)
    image_width = tf.cast(feature['image_width'],tf.int32)
    image_channel = tf.cast(feature['image_channel'],tf.int32)
    #print(image_height,image_width,image_channel)

    image_shape = tf.stack([image_height,image_width,image_channel])
    #image_data = tf.cast(image_data, tf.float32)
    image_data = tf.reshape(image_data, image_shape)

    image_label = tf.decode_raw(feature['image_label'],tf.float32)
    image_label = tf.reshape(image_label,[FLAGS.num_classes])

    #print image_data
    #print image_label
    return image_data,image_label
    pass

def main(unused_argv):
    with tf.Graph().as_default():
        #tf.logging.set_verbosity(tf.logging.DEBUG)
        image, label = parse_single_image(FLAGS.val_tfrecords)
        image.set_shape([None, None, 3])
        #image = tf.image.resize_images(image, [image_size, image_size])
        # Preprocess images
        image_pre = inception_preprocessing.preprocess_image(image, image_size, image_size, is_training=False)

        images, labels = tf.train.batch([image_pre, label], batch_size=FLAGS.batch_size, num_threads=2, capacity=10000)

        #input_images=tf.placeholder(tf.float32,shape=[FLAGS.batch_size,image_size,image_size,3])
        #targets=tf.placeholder(tf.float32,shape=[FLAGS.batch_size,FLAGS.num_classes])
        # create the model
        with tf.contrib.framework.arg_scope(inception_resnet_v2_arg_scope()):
            logits, _ = inception_resnet_v2(images, num_classes=FLAGS.num_classes, is_training=False)
        predictions = tf.nn.sigmoid(logits, name='predictions')
        tf.summary.histogram('predictions',predictions)
        cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(labels=labels, logits=logits)
        loss = tf.reduce_mean(cross_entropy)
        tf.summary.scalar('loss',loss)

        predict_labels = tf.cast(tf.greater_equal(predictions, FLAGS.threshold),tf.float32)
        correct_prediction = tf.equal(predict_labels, labels)
        correct_prediction = tf.cast(correct_prediction, tf.float32)
        total_accuracy = tf.reduce_mean(correct_prediction)
        tf.summary.scalar('total_validation_accuracy',total_accuracy)

        saver=tf.train.Saver()
        with tf.Session() as sess:
            #restore the inception_resnet_v2 model excepting the last layer
            saver.restore(sess, os.path.join(FLAGS.finetune_output_model_dir, FLAGS.checkpoint))
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(sess=sess, coord=coord)
            #val_steps_per_epoch = int(math.ceil(5377 / FLAGS.batch_size))
            val_steps_per_epoch = int(math.ceil(6436 / FLAGS.batch_size))
            val_total_accuracy = 0
            val_total_loss = 0
            prediction_list=[]
            label_list=[]
            val_count = 0
            # merge all the summaries
            merged_summary=tf.summary.merge_all()
            validation_summary_writer=tf.summary.FileWriter(os.path.join(FLAGS.multi_label_log_dir,'validation'))


            for i in range(val_steps_per_epoch):
                #image,image_pre=sess.run([image,image_pre])
                #print(image)
                #print(image_pre)
                #images_batch = sess.run(images)
                #labels_batch = sess.run(labels)
                #val_logits, val_predictions, val_predict_labels, val_loss, val_batch_accuracy = sess.run(
                 #   [logits, predictions, predict_labels, loss, total_accuracy],
                 #   feed_dict={input_images: images_batch,
                 #              targets: labels_batch})
                val_logits, val_predictions, val_predict_labels, val_labels, val_loss, val_batch_accuracy = sess.run([logits,predictions, predict_labels,labels, loss, total_accuracy])
                merged=sess.run(merged_summary)
                validation_summary_writer.add_summary(merged,i)
                #val_total_accuracy += val_batch_accuracy
                #val_count += val_predictions.shape[0]
                #val_total_loss += val_loss
                prediction_list.extend(val_predict_labels)
                label_list.extend(val_labels)

                print(val_predictions.shape)
                print('val_logits=%s' % val_logits)
                print('val_labels=%s' % val_labels)
                print('val_predictions=%s' % (val_predictions))
                print('val_predict_labels=%s' % (val_predict_labels))
                print('%s step=%d, val_loss=%.5f, val_batch_accuracy=%.5f' % (datetime.now(),i, val_loss, val_batch_accuracy * 100))

            print('np.array(prediction_list).shape={}'.format(np.array(prediction_list).shape))
            print('np.array(label_list).shape={}'.format(np.array(label_list).shape))
            #print(len(label_list))
            prediction_arr=np.array(prediction_list)
            label_arr=np.array(label_list)

            accuracy=(prediction_arr == label_arr)
            print('accuracy.shape={}'.format(accuracy.shape))
            print('total_accuracy=%.5f' % np.mean(accuracy))

            prediction_text=prediction_arr[:,19]
            label_text=label_arr[:,19]
            print('prediction_text=%s' % prediction_text)
            print('label_text=%s' % label_text)
            # compute the accuracy and the recall of text
            compute_text(prediction_text, label_text)

            coord.request_stop()
            coord.join(threads)

def compute_text(prediction_text, label_text):
    text_accuracy=(prediction_text == label_text)
    text_accuracy=np.mean(text_accuracy)
    text_count=0
    correct_text_count=0
    for index, value in enumerate(prediction_text):
        label = label_text[index]
        if(label == 1):
            text_count += 1
            if(value == 1):
                correct_text_count += 1

    text_recall=(float(correct_text_count)/text_count)*100
    print('text_accuracy={0}%,text_recall={1:.5f}%'.format(text_accuracy*100,text_recall))



if __name__=='__main__':
    tf.app.run()
