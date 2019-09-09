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
        self.cls_acc = self.net.compute_accuracy(self.cls_target, cls_pred)
        self.train = self.optimizer.minimize(self.net.evaluate_loss(self.cls_loss, self.bbox_loss))

        

    def train(self, batch_size=32, n_epoch=10):
        self.define_training_var()
        iterator = self.dataset.make_initializable_iterator()
        next_element = iterator.get_next()
        with tf.Session() as sess:
            init_global = tf.compat.v1.initializers.global_variables()
            init_local = tf.compat.v1.initializers.local_variables()
            sess.run(init_global)
            sess.run(init_local)
            for e in range(10):
                sess.run(iterator.initializer)
                faces_losses = []
                bbox_losses = []
                cls_accs = []
                while True:
                    try:
                        image, label, roi = sess.run(next_element)
                        _, face_loss_val, bbox_loss_val, cls_acc_val = sess.run((self.train, self.cls_loss, self.bbox_loss, self.cls_acc), feed_dict={self.x_input: image, self.cls_target: label, self.bbox_target: roi})
                        faces_losses.append(face_loss_val)
                        bbox_losses.append(bbox_loss_val)
                        cls_accs.append(cls_acc_val)
                    except tf.errors.OutOfRangeError:
                        break
                print("Epoch {}, face_loss:{}, bbox_loss:{}, cls_accuracy:{}".format(e, np.mean(np.array(faces_losses)), np.mean(np.array(bbox_losses)),np.mean(np.array(cls_accs))))