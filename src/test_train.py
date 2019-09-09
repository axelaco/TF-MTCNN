import tensorflow as tf
from mtcnn_models import PNet
from trainer import Trainer
import numpy as np




def parser(record):
    image_feature_description = {
        'image/raw': tf.FixedLenFeature([], tf.string),
        'image/height': tf.FixedLenFeature([], tf.int64),
        'image/width': tf.FixedLenFeature([], tf.int64),
        'image/channels': tf.FixedLenFeature([], tf.int64),
        'image/object/bbox/label':tf.FixedLenFeature([], tf.int64),
        'image/object/bbox/xmin':tf.FixedLenFeature([], tf.float32),
        'image/object/bbox/xmax':tf.FixedLenFeature([], tf.float32),
        'image/object/bbox/ymin':tf.FixedLenFeature([], tf.float32),
        'image/object/bbox/ymax':tf.FixedLenFeature([], tf.float32)
    }
    image_feature = tf.parse_single_example(record, image_feature_description)


    # Perform additional preprocessing on the parsed data.
    height = image_feature['image/height']
    width = image_feature['image/width']
    channels = image_feature['image/channels']
    image = tf.decode_raw(image_feature['image/raw'], tf.uint8)
    image = tf.reshape(image, [height, width, channels])
    image = (tf.cast(image, tf.float32)-127.5) / 128
    label = (tf.cast(image_feature['image/object/bbox/label'], tf.float32))
    label = tf.reshape(label, [1])
    xmin = image_feature['image/object/bbox/xmin']
    xmax = image_feature['image/object/bbox/xmax']
    ymin = image_feature['image/object/bbox/ymin']
    ymax = image_feature['image/object/bbox/ymax']
    roi = [xmin, ymin, xmax, ymax]
    return image, label, roi



pnet = PNet(alpha_class=1, beta_reg=0.5, class_th=0.5)


n_epoch = 1
batch_size = 32

filenames = ["/home/axel/Documents/MTCNN/data/12/tf_ann.tfrecords"]
dataset = tf.data.TFRecordDataset(filenames)
dataset = dataset.map(parser)
dataset = dataset.shuffle(buffer_size=10000)
dataset = dataset.batch(32)
dataset = dataset.repeat(n_epoch)

optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=1e-4)
trainer = Trainer(net=pnet, dataset=dataset, optimizer=optimizer)
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