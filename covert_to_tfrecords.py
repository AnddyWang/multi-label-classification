#written by pengkai.wang
import os
import tensorflow as tf

os.environ['CUDA_VISIBLE_DEVICES']='0'

tf.app.flags.DEFINE_string('data_dir','/data0/users/pengkai1/datasets/MultiLabel','image_list')
tf.app.flags.DEFINE_string('labels_file','labels.txt','labels_file')

FLAGS=tf.app.flags.FLAGS

def parse_label_to_list(label,labels_file):
    pass


def convert_to_tfrecords(image_list,label_list,labels_file):
    pass

def main():
    convert_to_tfrecords(image_list, label_list, labels_file)

if __name__=='__main__':
    tf.app.run()