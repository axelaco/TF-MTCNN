import tensorflow as tf
from base_model import MTCNNNetwork

class PNet(MTCNNNetwork):
    
    def __init__(self):
        super(PNet, self).__init__(name='PNet', alpha_class=1.0, beta_reg=0.5, class_th=0.5)
        # Define your layers here.
        self.conv1 = tf.keras.layers.Conv2D(filters=10, kernel_size=(3,3), strides=1, input_shape=(12, 12, 3))
        self.max_pool_conv1 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=2, padding='SAME')
        self.prelu1 = tf.keras.layers.PReLU()
        self.conv2 = tf.keras.layers.Conv2D(filters=16, kernel_size=(3,3), strides=1)
        self.prelu2 = tf.keras.layers.PReLU()
        self.conv3 = tf.keras.layers.Conv2D(filters=32, kernel_size=(3,3), strides=1)
        self.prelu3 = tf.keras.layers.PReLU()
        self.conv4_1 = tf.keras.layers.Conv2D(filters=1, kernel_size=1, strides=1)
        self.conv4_2 = tf.keras.layers.Conv2D(filters=4, kernel_size=(1, 1), strides=1)

    def call(self, inputs, training=False):
        # Define your forward pass here,
        # using layers you previously defined (in `__init__`).
        x = self.prelu1(self.max_pool_conv1(self.conv1(inputs)))
        x = self.prelu2(self.conv2(x))
        x = self.prelu3(self.conv3(x))

        y_face_pred = tf.keras.activations.sigmoid(self.conv4_1(x))
        y_bbox_pred = self.conv4_2(x)

        return y_face_pred, y_bbox_pred

    def compute_loss(self, target_class, target_reg, pred_class, pred_reg):    
        return super().compute_loss(target_class, target_reg, pred_class, pred_reg, squeeze=True)

    def compute_accuracy(self, target, pred):
        return super().compute_accuracy(target, pred, squeeze=True)


class RNet(MTCNNNetwork):
    def __init__(self):
        super(RNet, self).__init__(name='RNet', alpha_class=1.0, beta_reg=0.5, class_th=0.5)
        
        self.conv1 = tf.keras.layers.Conv2D(filters=28, kernel_size=(3,3), strides=1, input_shape=(24, 24, 3))
        self.max_pool_conv1 = tf.keras.layers.MaxPooling2D(pool_size=(3, 3), strides=2, padding='SAME')
        self.prelu1 = tf.keras.layers.PReLU()
        self.conv2 = tf.keras.layers.Conv2D(filters=48, kernel_size=(3,3), strides=1)
        self.max_pool_conv2 = tf.keras.layers.MaxPooling2D(pool_size=(3, 3), strides=2, padding='VALID')
        self.prelu2 = tf.keras.layers.PReLU()
        self.conv3 = tf.keras.layers.Conv2D(filters=64, kernel_size=(2,2), strides=1)
        self.prelu3 = tf.keras.layers.PReLU()
        self.fc1 = tf.keras.layers.Dense(units=128)
        self.fc_prelu = tf.keras.layers.PReLU()
        
        self.fc_face = tf.keras.layers.Dense(units=1)

        self.fc_bbox = tf.keras.layers.Dense(units=4)
        

    def call(self, inputs, training=False):
        x = self.prelu1(self.max_pool_conv1(self.conv1(inputs)))
        
        x = self.prelu2(self.max_pool_conv2(self.conv2(x)))
        
        x = self.prelu3(self.conv3(x))

        x_base = self.fc_prelu(self.fc1(tf.keras.layers.Flatten()(x)))
       
        y_face = tf.keras.activations.sigmoid(self.fc_face(x_base))
        y_bbox = self.fc_bbox(x_base)

        return y_face, y_bbox