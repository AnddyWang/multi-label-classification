#written by pengkai.wang
import os
import tensorflow as tf
import numpy as np

os.environ['CUDA_VISIBLE_DEVICES']='1'

tf.app.flags.DEFINE_string('data_dir','/data0/users/pengkai1/datasets/MultiLabel','image_list')
tf.app.flags.DEFINE_string('train_tfrecords','/data0/users/pengkai1/datasets/MultiLabel/train.tfrecords','train_tfrecords')
tf.app.flags.DEFINE_string('test_tfrecords','/data0/users/pengkai1/datasets/MultiLabel/test.tfrecords','test_tfrecords')
tf.app.flags.DEFINE_string('labels_file','labels.txt','labels_file')
tf.app.flags.DEFINE_string('output_dir', '/data0/users/pengkai1/datasets/MultiLabel', 'output dir')
tf.app.flags.DEFINE_integer('num_classes',14,'the number of classes')

FLAGS=tf.app.flags.FLAGS

def parse_label_to_list(label_list,labels_file):
    pass
    labels_list_file = open(os.path.join(FLAGS.output_dir, label_list), 'r')
    all_labels_file = open(os.path.join(FLAGS.output_dir, labels_file), 'r')
    all_labels=[line.strip() for line in all_labels_file.readlines()]

    labels=[]
    for line in labels_list_file.readlines():
        encoded_label=np.zeros(FLAGS.num_classes,np.float32)
        line=line.strip()
        parts=line.split(' ')
        for part in parts:
            encoded_label[get_label_index(part,all_labels)]=1.0
        labels.append(encoded_label.tostring())

#get the index of label in the all labels list
def get_label_index(label,labels):
    label_index = -1
    for index , value in labels:
        if label == value:
            label_index = index
            break
    return label_index

def convert_to_tfrecords(category,image_list_file,label_list_file,labels_file):
    image_file=open(os.path.join(FLAGS.output_dir,image_list_file),'r')
    labels=parse_label_to_list(label_list_file,labels_file)
    #write the data into tfrecords file
    with tf.python_io.TFRecordWriter(category) as tf_writer:
        for index,value in enumerate([image_list.strip() for image_list in image_file.readlines()]):
            label=labels[index]
            image_data=tf.gfile.FastGFile(os.path.join(FLAGS.data_dir,'images',value),'r').read()

            #use the example proto
            example=tf.train.Example(features=tf.train.Features(feature={
                'image_data' : tf.train.Feature(bytes_list=tf.train.BytesList(value=[image_data])),
                'image_label' : tf.train.Feature(bytes_list=tf.train.BytesList(value=[label]))
            }))
            tf_writer.write(example.SerializeToString())


def main():
    convert_to_tfrecords(FLAGS.train_tfrecords,'train_lists.txt', 'train_labels.txt', FLAGS.labels_file)
    convert_to_tfrecords(FLAGS.test_tfrecords,'test_lists.txt', 'test_labels.txt', FLAGS.labels_file)

if __name__=='__main__':
    tf.app.run()