import tensorflow as tf

def filter_negative_samples(labels, tensors):
    keeps_indices = tf.where(tf.math.greater_equal(labels, 0))
    keeps_indices = keeps_indices[:,0]
    
    filtered = []
    for t in tensors:
        with tf.control_dependencies([tf.compat.v1.debugging.assert_equal(tf.shape(t)[0], tf.shape(labels)[0])]):
            f = tf.gather(t, keeps_indices)
            filtered.append(f)
    
    return filtered

def filter_zeros_samples(labels, tensors):
    keeps_indices = tf.where(tf.math.not_equal(labels, 0))
    keeps_indices = keeps_indices[:,0]
    
    filtered = []
    for t in tensors:
        with tf.control_dependencies([tf.compat.v1.debugging.assert_equal(tf.shape(t)[0], tf.shape(labels)[0])]):
            f = tf.gather(t, keeps_indices)
            filtered.append(f)
    
    return filtered