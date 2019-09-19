import tensorflow as tf
from tf_utils_func import filter_negative_samples, filter_zeros_samples
from abc import abstractmethod

class MTCNNNetwork(tf.keras.Model):
    def __init__(self, name, alpha_class, beta_reg, class_th):
        super(MTCNNNetwork, self).__init__(name='name')
        self.acc = tf.keras.metrics.Accuracy()
        self.alpha_class = alpha_class
        self.beta_reg = beta_reg
        self.class_th = class_th


    def evaluate_loss(self, cls_loss, reg_loss):
        return self.alpha_class * cls_loss + self.beta_reg * reg_loss

    
    def compute_loss(self, target_class, target_reg, pred_class, pred_reg, squeeze=False):
        if squeeze:
            pred_class = tf.squeeze(pred_class, [1, 2], name='cls_prob')
            pred_reg = tf.squeeze(pred_reg, [1, 2], name='cls_bbox')
            
        filtered_face = filter_negative_samples(labels=target_class, tensors=[target_class, pred_class])

        filtered_bbox = filter_zeros_samples(labels=target_class, tensors=[target_reg, pred_reg])

        face_loss = tf.reduce_mean(tf.keras.losses.binary_crossentropy(y_true=filtered_face[0], y_pred=filtered_face[1]))
        bbox_loss = tf.reduce_mean(tf.compat.v1.losses.huber_loss(labels=filtered_bbox[0], predictions=filtered_bbox[1]))

        return face_loss, bbox_loss

    def compute_accuracy(self, target, pred, squeeze=False):
        if squeeze:
            pred = tf.squeeze(pred, [1, 2], name='cls_prob')
        
        filtered_face = filter_negative_samples(labels=target, tensors=[target, pred])


        ones = tf.ones_like(filtered_face[1], dtype=tf.int32)
        
        mask = tf.math.greater(filtered_face[1], 0.5)
        mask = tf.cast(mask, dtype=tf.int32)
        y_pred_met = tf.multiply(ones, tf.cast(mask, dtype=tf.int32))
        label_casted = tf.cast(filtered_face[0], dtype=tf.int32)
        self.acc.update_state(label_casted, y_pred_met)
        return self.acc.result()
    
    @abstractmethod
    def call(self, input, trainable=True):
        pass