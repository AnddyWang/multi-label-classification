import tensorflow as tf
import os

os.environ['CUDA_VISIBLE_DEVICES']='0'

graph=tf.get_default_graph()

with tf.Session() as sess:
    restore = tf.train.import_meta_graph('model-1000.meta')
    restore.restore(sess,tf.train.latest_checkpoint('./'))

    w1=graph.get_tensor_by_name('w1:0')
    w2=graph.get_tensor_by_name('w2:0')
    b1=graph.get_tensor_by_name('b1:0')
    #print(sess.run([b1]))
    feed={w1:15,w2:17}
    print(sess.run([w1,w2,b1],feed_dict=feed))
    op_to_restore=graph.get_tensor_by_name('op_to_restore:0')
    print(sess.run(op_to_restore,feed_dict=feed))
    add_op =tf.multiply(op_to_restore,2)
    print(sess.run(add_op, feed_dict=feed))

