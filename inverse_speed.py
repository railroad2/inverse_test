import numpy as np
import tensorflow as tf
import time

mat = np.array(np.random.randn(6000, 6000))#, dtype=np.float32)

def test_tensorflow():

    X = tf.constant(mat)#, dtype=np.float32)
    Xi = tf.linalg.inv(X)

    init = tf.initialize_all_variables()
    sess = tf.Session()
    sess.run(init)

    t0 = time.time()
    Xi = sess.run(Xi)
    print (time.time() - t0)
    print (Xi)

def test_numpy():
    t0 = time.time()
    Xi = np.linalg.inv(mat)
    print (time.time() - t0)
    print (Xi)

if __name__=="__main__":
    #test_tensorflow()
    test_numpy()
