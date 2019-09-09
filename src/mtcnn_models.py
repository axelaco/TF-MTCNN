from tf_utils_func import filter_negative_samples, filter_zeros_samples
from network import Network
import tensorflow as tf


class PNet(Network):
    def __init__(self, alpha_class, beta_reg, class_th):
        super().__init__(alpha_class, beta_reg, class_th)

        self.conv1 = tf.keras.layers.Conv2D(filters=10, kernel_size=(3,3), strides=1)

        self.conv2 = tf.keras.layers.Conv2D(filters=16, kernel_size=(3,3), strides=1)

        self.conv3 = tf.keras.layers.Conv2D(filters=32, kernel_size=(3,3), strides=1)

        self.conv4_1 = tf.keras.layers.Conv2D(filters=1, kernel_size=1, strides=1)

        self.conv4_2 = tf.keras.layers.Conv2D(filters=4, kernel_size=(1, 1), strides=1)

    def forward(self, input, trainable=True):
        x = self.conv1(input)
        x = tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=2, padding='SAME')(x)
        x = tf.keras.layers.PReLU()(x)
        
        print(x.get_shape())
        x = self.conv2(x)
        x = tf.keras.layers.PReLU()(x)

        print(x.get_shape())
        
        x = self.conv3(x)
        x = tf.keras.layers.PReLU()(x)

        print(x.get_shape())        
        y_face_pred = tf.keras.activations.sigmoid(self.conv4_1(x))

        y_bbox_pred = self.conv4_2(x)

        print(y_bbox_pred.get_shape())
        print(y_face_pred.get_shape())


        return y_face_pred, y_bbox_pred
    

    def compute_loss(self, target_class, target_reg, pred_class, pred_reg):
        
        pred_class = tf.squeeze(pred_class, [1, 2], name='cls_prob')
        pred_reg = tf.squeeze(pred_reg, [1, 2], name='cls_bbox')
        
        filtered_face = filter_negative_samples(labels=target_class, tensors=[target_class, pred_class])

        filtered_bbox = filter_zeros_samples(labels=target_class, tensors=[target_reg, pred_reg])

        face_loss = tf.reduce_mean(tf.keras.losses.binary_crossentropy(y_true=filtered_face[0], y_pred=filtered_face[1]))
        bbox_loss = tf.reduce_mean(tf.compat.v1.losses.huber_loss(labels=filtered_bbox[0], predictions=filtered_bbox[1]))

        return face_loss, bbox_loss

    def get_input(self):
        return tf.compat.v1.placeholder(tf.float32, shape=(None, 12, 12, 3))
    
    def get_target_outputs(self):
        cls_target = tf.compat.v1.placeholder(tf.float32, shape=(None, 1))
        bbox_target = tf.compat.v1.placeholder(tf.float32, shape=(None, 4))
        return cls_target, bbox_target
    
    def evaluate_loss(self, cls_loss, reg_loss):
        return self.alpha_class * cls_loss + self.beta_reg * reg_loss