import tensorflow as tf
import numpy as np

class Trainer(object):
    def __init__(self, net, dataset, optimizer):
        self.net = net
        self.optimizer = optimizer
        self.dataset = dataset
    
    
    def define_training_var(self):
        self.x_input = self.net.get_input()
        self.cls_target, self.bbox_target = self.net.get_target_outputs()
        cls_pred, bbox_pred = self.net.forward(self.x_input, trainable=True)
        self.cls_loss, self.bbox_loss = self.net.compute_loss(self.cls_target, self.bbox_target, cls_pred, bbox_pred)
        self.train = self.optimizer.minimize(self.net.evaluate_loss(self.cls_loss, self.bbox_loss))

        

    def train(self, batch_size=32, n_epoch=10):
        self.define_training_var()
        with tf.compat.v1.Session() as sess:
            # Init session
            init_global = tf.compat.v1.initializers.global_variables()
            init_local = tf.compat.v1.initializers.local_variables()
            sess.run(init_global)
            sess.run(init_local)

            for epoch in range(0, n_epoch):
                x_input = np.random.rand(batch_size, 12, 12, 3)
                y_cls = np.random.randint(-1, 2, size=(batch_size, 1))
                y_bbox = np.random.rand(batch_size, 4)

                _, face_loss_val, bbox_loss_val = sess.run((self.train, self.cls_loss, self.bbox_loss), feed_dict={self.x_input: x_input, self.cls_target: y_cls, self.bbox_target: y_bbox})

                print(face_loss_val, bbox_loss_val)
        