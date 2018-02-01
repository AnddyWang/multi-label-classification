#written by pengkai.wang
import os
import tensorflow as tf
import numpy as np
import math
from PIL import Image

os.environ['CUDA_VISIBLE_DEVICES']='1'

tf.app.flags.DEFINE_string('data_dir','/data0/users/pengkai1/datasets/MultiLabelImage','image_list')
tf.app.flags.DEFINE_string('train_tfrecords','/data0/users/pengkai1/datasets/MultiLabelImage/train.tfrecords','train_tfrecords')
tf.app.flags.DEFINE_string('test_tfrecords','/data0/users/pengkai1/datasets/MultiLabelImage/test.tfrecords','test_tfrecords')
tf.app.flags.DEFINE_string('labels_file','data/labels.txt','labels_file')
tf.app.flags.DEFINE_string('output_dir', '/data0/users/pengkai1/datasets/MultiLabelImage', 'output dir')
tf.app.flags.DEFINE_integer('num_classes',21,'the number of classes')

FLAGS=tf.app.flags.FLAGS

def parse_label_to_list(label_file):
    label_list_file = open(label_file, 'r')
    all_labels_file = open(FLAGS.labels_file, 'r')
    all_labels=[line.strip() for line in all_labels_file.readlines()]

    encoded_label = np.zeros(FLAGS.num_classes, np.float32)
    for line in label_list_file.readlines():
        line=line.strip()
        encoded_label[get_label_index(line, all_labels)] = 1.0
    print('encoded_label={}'.format(encoded_label))
    return encoded_label.tostring()

#get the index of label in the all labels list
def get_label_index(label, all_labels):
    label_index = -1
    for index , value in enumerate(all_labels):
        if label == value:
            label_index = index
            break
    return label_index

def convert_to_tfrecords(category,image_list):
    #write the data into tfrecords file
    with tf.python_io.TFRecordWriter(category) as tf_writer:
        for image_file in image_list:
            label_file = os.path.splitext(image_file)[0]+'.txt'
            image_raw = Image.open(image_file)
            image_shape = np.array(image_raw).shape
            image_data = image_raw.tobytes()
            label = parse_label_to_list(label_file)

            # use the example proto
            example = tf.train.Example(features=tf.train.Features(feature={
                'image_data': tf.train.Feature(bytes_list=tf.train.BytesList(value=[image_data])),
                'image_height': tf.train.Feature(int64_list=tf.train.Int64List(value=[image_shape[0]])),
                'image_width': tf.train.Feature(int64_list=tf.train.Int64List(value=[image_shape[1]])),
                'image_channel': tf.train.Feature(int64_list=tf.train.Int64List(value=[image_shape[2]])),
                'image_label': tf.train.Feature(bytes_list=tf.train.BytesList(value=[label]))
            }))
            tf_writer.write(example.SerializeToString())


def main(unused_argv):
    # store the image list
    jpgFileList = []
    for currentDirPath, subDirPath, fileList in os.walk(FLAGS.data_dir):
        for index, fileName in enumerate(fileList):
            if fileName.endswith('.jpg'):
                print(os.path.join(currentDirPath,fileName))
                jpgFileList.append(os.path.join(currentDirPath,fileName))

    image_count = len(jpgFileList)
    print('Jpg Count={}'.format(image_count))
    # set 30% for the test
    test_image_count = int(math.ceil(image_count*0.3))
    random_index = np.random.permutation(image_count)
    train_image_index = random_index[test_image_count:]
    test_image_index = random_index[:test_image_count]
    jpgFileList=np.array(jpgFileList)
    train_image_list=jpgFileList[train_image_index]
    test_image_list = jpgFileList[test_image_index]
    print('train_image_count={},test_image_count={}'.format(len(train_image_list),len(test_image_list)))
    convert_to_tfrecords(FLAGS.train_tfrecords, train_image_list)
    convert_to_tfrecords(FLAGS.test_tfrecords, test_image_list)

if __name__=='__main__':
    tf.app.run()