import tensorflow as tf
tf.enable_eager_execution()
import numpy as np
import tqdm
import datetime

def count_nb(X, val):
    return tf.reduce_sum(tf.cast(tf.equal(X, val), tf.int32))

def filter_element(val, idx_max, label, tensors):
    keeps_indices = tf.where(tf.math.equal(label, val))
    keeps_indices = keeps_indices[:idx_max, 0]
    filtered = []
    for t in tensors:
        with tf.control_dependencies([tf.compat.v1.debugging.assert_equal(tf.shape(t)[0], tf.shape(label)[0])]):
            f = tf.gather(t, keeps_indices)
            filtered.append(f)
    return filtered

def rebalance_batch(x, label, roi):
        pos = count_nb(label, 1)

        x_neg, label_neg, roi_neg = filter_element(0, pos * 2, label, [x, label, roi])

        x_part, label_part, roi_part  = filter_element(2, pos, label, [x, label, roi])

        x_pos, label_pos, roi_pos = filter_element(1, pos, label, [x, label, roi])

        part = count_nb(label, 2)
        
        if part < pos:
            return None, None, None

        indexes = tf.range(0, 4 * pos + 1)
    
        indexes = tf.random.shuffle(indexes)
    
        x_final = tf.gather(tf.concat([x_neg, x_part, x_pos], axis=0), indexes)
        label_final = tf.gather(tf.concat([label_neg, label_part, label_pos], axis=0), indexes)
        roi_final = tf.gather(tf.concat([roi_neg, roi_part, roi_pos], axis=0), indexes)        

        return x_final, label_final, roi_final


class Trainer(object):
    def __init__(self, net, train_dataset, val_dataset, optimizer):
        # Init Network hyperparameter 
        self.net = net
        self.optimizer = optimizer
        self.dataset = train_dataset
        self.val_dataset = val_dataset

        # Init the checkpoint model saver
        self.ckpt = tf.train.Checkpoint(step=tf.Variable(1), optimizer=self.optimizer, net=self.net)
        self.manager = tf.train.CheckpointManager(self.ckpt, './tf_ckpts/{}'.format(self.net.__class__.__name__), max_to_keep=None)


        # Init the metrics variable to follow the training
        self.loss_metric = tf.keras.metrics.Mean(name='train_loss')
        self.bbox_loss_metric = tf.keras.metrics.Mean(name='bbox_train_loss')
        self.cls_accuracy = tf.keras.metrics.Mean(name='train_cls_acc')

        self.val_loss_metric = tf.keras.metrics.Mean(name='val_loss')
        self.val_bbox_loss_metric = tf.keras.metrics.Mean(name='bbox_val_loss')
        self.val_cls_accuracy = tf.keras.metrics.Mean(name='val_cls_acc')

        # Init the tensorboard writer
        current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        self.train_log_dir = 'logs/gradient_tape/' + self.net.__class__.__name__ + '/' + current_time + '/train'
        self.val_log_dir = 'logs/gradient_tape/' + self.net.__class__.__name__ + '/' + current_time + '/val'
        self.train_summary_writer = tf.contrib.summary.create_file_writer(self.train_log_dir)
        self.val_summary_writer = tf.contrib.summary.create_file_writer(self.val_log_dir)

    @tf.function
    def train_step(self, inputs, labels, rois):
        with tf.GradientTape() as tape:
            cls_pred, bbox_pred = self.net(inputs, training=True)
            face_loss, bbox_loss = self.net.compute_loss(labels, rois, cls_pred, bbox_pred)
            acc = self.net.compute_accuracy(labels, cls_pred)
            total_loss = self.net.evaluate_loss(face_loss, bbox_loss)

        gradients = tape.gradient(total_loss, self.net.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.net.trainable_variables))
        self.loss_metric.update_state(face_loss)
        self.bbox_loss_metric.update_state(bbox_loss)
        self.cls_accuracy.update_state(acc)


    @tf.function
    def val_step(self, inputs, labels, rois):
        with tf.GradientTape() as tape:
            cls_pred, bbox_pred = self.net(inputs, training=True)
            face_loss, bbox_loss = self.net.compute_loss(labels, rois, cls_pred, bbox_pred)
            acc = self.net.compute_accuracy(labels, cls_pred)
            
        self.val_loss_metric.update_state(face_loss)
        self.val_bbox_loss_metric.update_state(bbox_loss)
        self.val_cls_accuracy.update_state(acc)

    def save_model(self):
        self.ckpt.step.assign_add(1)
        save_path = self.manager.save()
        print("Saved checkpoint for step {}: {}".format(int(self.ckpt.step), save_path))


    def log_metrics(self, n_epoch):
        mean_loss = self.loss_metric.result()
        mean_bbox_loss = self.bbox_loss_metric.result()
        mean_acc = self.cls_accuracy.result()

        val_mean_loss = self.val_loss_metric.result()
        val_mean_bbox_loss = self.val_bbox_loss_metric.result()
        val_mean_acc = self.val_cls_accuracy.result()
        
        with self.train_summary_writer.as_default(),  tf.contrib.summary.always_record_summaries():
            tf.contrib.summary.scalar('face_loss', mean_loss, step=n_epoch)
            tf.contrib.summary.scalar('bbox_loss', mean_bbox_loss, step=n_epoch)
            tf.contrib.summary.scalar('cls_acc', mean_acc, step=n_epoch)


        with self.val_summary_writer.as_default(),  tf.contrib.summary.always_record_summaries():
            tf.contrib.summary.scalar('face_loss', val_mean_loss, step=n_epoch)
            tf.contrib.summary.scalar('bbox_loss', val_mean_bbox_loss, step=n_epoch)
            tf.contrib.summary.scalar('cls_acc', val_mean_acc, step=n_epoch)

        print('Epoch {}: cls_loss={:.3f} bbox_loss={:.3f} cls_acc={:.3f}'.format(n_epoch, mean_loss, mean_bbox_loss, mean_acc))
        print('val_cls_loss={:.3f} val_bbox_loss={:.3f} val_cls_acc={:.3f}'.format(val_mean_loss, val_mean_bbox_loss, val_mean_acc))


    
    def train(self, n_epoch, mini_batch=False):
        for epoch in range(n_epoch):
            self.loss_metric.reset_states()
            self.bbox_loss_metric.reset_states()
            self.cls_accuracy.reset_states()

            self.val_loss_metric.reset_states()
            self.val_bbox_loss_metric.reset_states()
            self.val_cls_accuracy.reset_states()
            
            for x, label, roi in tqdm.tqdm(self.dataset):
                if mini_batch:
                    x, label, roi = rebalance_batch(x, label, roi)
                    if x == None:
                        continue
                    
                self.train_step(x, label, roi)
            
            for inputs, label, roi in tqdm.tqdm(self.val_dataset):
                self.val_step(inputs, label, roi)
                break
            
            self.save_model()

            self.log_metrics(epoch)
