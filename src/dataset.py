import tensorflow as tf
import glob

class MTCNNDataset(object):
    def __init__(self, path_to_datasets):

        self.val_set_path = glob.glob(path_to_datasets + '*_val.tfrecords')

        self.train_set_path = glob.glob(path_to_datasets + '*_train.tfrecords')

    def parser(self, record):
    
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


    def get_training_set(self, batch_size=64):
        filenames = [self.train_set_path]
        dataset = tf.data.TFRecordDataset(filenames)
        dataset = dataset.repeat(1)
        dataset = dataset.shuffle(buffer_size=10000)
        dataset = dataset.apply(tf.contrib.data.map_and_batch(
            map_func=self.parser, batch_size=batch_size))
        return dataset
    
    def get_validation_set(self, batch_size=64):
        filenames = [self.val_set_path]
        dataset = tf.data.TFRecordDataset(filenames)
        dataset = dataset.repeat(1)
        dataset = dataset.shuffle(buffer_size=10000)
        dataset = dataset.apply(tf.contrib.data.map_and_batch(
            map_func=self.parser, batch_size=batch_size))
        return dataset