import tensorflow as tf

def parse_single_image(tfrecords_file):
    file_queue=tf.train.string_input_producer([tfrecords_file])
    reader=tf.TFRecordReader()
    filename,serialized_example=reader.read(file_queue)
    #print('filename=%s,serialized_example=%s',filename,serialized_example)
    feature=tf.parse_single_example(serialized_example,features={
        'image_data':tf.FixedLenFeature([], tf.string),
        'image_shape':tf.FixedLenFeature([], tf.string),
        'image_label':tf.FixedLenFeature([], tf.string)
    })

    image_data = tf.decode_raw(feature['image_data'], tf.uint8)
    image_shape = tf.decode_raw(feature['image_shape'], tf.int32)
    image_data = tf.reshape(image_data, image_shape)

    image_label = tf.decode_raw(feature['image_label'], tf.float32)
    image_label = tf.reshape(image_label, [FLAGS.num_classes])

    return image_data, image_label, image_shape