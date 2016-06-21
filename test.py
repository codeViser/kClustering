import tensorflow as tf 
import numpy as np

x=tf.Variable(np.array([2.3465465,3.564650,4.563164]))
y=tf.Variable(np.array([1.,2.,3.]))
copy_x=y.assign(x)
model = tf.initialize_all_variables()

with tf.Session() as session:
	session.run(model)
	a=session.run(x)
	b=session.run(copy_x)
	print(a)
	print(b)
