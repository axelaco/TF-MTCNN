import tensorflow as tf
import numpy as np
import tqdm
import datetime

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


    def train(self, n_epoch):
        for epoch in range(n_epoch):
            self.loss_metric.reset_states()
            self.bbox_loss_metric.reset_states()
            self.cls_accuracy.reset_states()

            self.val_loss_metric.reset_states()
            self.val_bbox_loss_metric.reset_states()
            self.val_cls_accuracy.reset_states()
            
            for inputs, label, roi in tqdm.tqdm(self.dataset):
                self.train_step(inputs, label, roi)

            for inputs, label, roi in tqdm.tqdm(self.val_dataset):
                self.val_step(inputs, label, roi)

            self.save_model()

            self.log_metrics(epoch)