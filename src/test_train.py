import tensorflow as tf
from network import PNet
from trainer import Trainer
import numpy as np

pnet = PNet(alpha_class=1, beta_reg=0.5, class_th=0.5)
optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=1e-4)
trainer = Trainer(net=pnet, dataset=None, optimizer=optimizer)
trainer.train()
"""
batch_size = 32
n_epoch = 10
shape = 12
input_image = tf.compat.v1.placeholder(tf.float32, shape=(None, shape, shape, 3))
label_gt = tf.compat.v1.placeholder(tf.float32, shape=(None, 1))
label_bbox_gt = tf.compat.v1.placeholder(tf.float32, shape=(None, 4))


y_face_pred, y_bbox_pred = pnet.forward(input_image, trainable=True)
cls_loss, bbox_loss = pnet.compute_loss(label_gt, label_bbox_gt, y_face_pred, y_bbox_pred)


optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=1e-4)
train = optimizer.minimize(1.0 * cls_loss + 0.5 * bbox_loss)



with tf.compat.v1.Session() as sess:
        # Init session
        init_global = tf.compat.v1.initializers.global_variables()
        init_local = tf.compat.v1.initializers.local_variables()
        sess.run(init_global)
        sess.run(init_local)

        for epoch in range(0, n_epoch):
            x_input = np.random.rand(batch_size, shape, shape, 3)
            y_cls = np.random.randint(-1, 2, size=(batch_size, 1))
            y_bbox = np.random.rand(batch_size, 4)

            _, face_loss_val, bbox_loss_val = sess.run((train, cls_loss, bbox_loss), feed_dict={input_image: x_input, label_gt: y_cls, label_bbox_gt: y_bbox})

            print(face_loss_val, bbox_loss_val)
"""