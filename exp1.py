import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
#building computational graph
w=tf.Variable([2.5,-0.2,1.0],tf.float32)
b=tf.Variable([-0.5],tf.float32)
x=tf.placeholder(tf.float32)
u=tf.tensordot(w,x,axes=1)+b
y=0.8/(1+tf.exp(-1.2*u))
sess=tf.Session()
init=tf.global_variables_initializer()
sess.run(init)
u,y=sess.run([u,y],{x:[0.8,2.0,-0.5]})
print(u,y)
