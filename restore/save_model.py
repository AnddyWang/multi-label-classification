import tensorflow as tf
import os

os.environ['CUDA_VISIBLE_DEVICES']='0'

w1=tf.placeholder('float',name='w1')
w2=tf.placeholder('float',name='w2')
b1=tf.Variable(2.0,name="b1")

w3=tf.add(w1,w2)
w4=tf.multiply(w3,b1,name='op_to_restore')

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    saver=tf.train.Saver()
    sess.run(w4,feed_dict={w1:3,w2:4})
    saver.save(sess,'./model',global_step=1000)
