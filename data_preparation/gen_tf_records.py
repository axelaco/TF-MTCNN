import tensorflow as tf
import tqdm
import sys
import numpy as np
import cv2

def _bytes_feature(value):
  """Returns a bytes_list from a string / byte."""
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _float_feature(value):
  """Returns a float_list from a float / double."""
  return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))

def _int64_feature(value):
  """Returns an int64_list from a bool / enum / int / uint."""
  return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

DATA_GENERATED_FOLDER = '/home/axel/Documents/MTCNN/data/'

def _convert_to_example(image_path, cls_target, bbox_target, net_size):
    img = cv2.imread(image_path)
    h, w, c = img.shape
    if h != net_size or w != net_size:
        img = cv2.resize(img, (net_size, net_size))
    img = img.astype('uint8')
    image_string = img.tostring()
    shape = img.shape
    example = tf.train.Example(features=tf.train.Features(feature={
        'image/raw': _bytes_feature(image_string),
        'image/height': _int64_feature(shape[0]),
        'image/width': _int64_feature(shape[1]),
        'image/channels': _int64_feature(shape[2]),
        'image/object/bbox/label': _int64_feature(int(cls_target)),
        'image/object/bbox/xmin':_float_feature(float(bbox_target[0])),
        'image/object/bbox/xmax':_float_feature(float(bbox_target[2])),
        'image/object/bbox/ymin':_float_feature(float(bbox_target[1])),
        'image/object/bbox/ymax':_float_feature(float(bbox_target[3]))
    }))
    return example.SerializeToString()

def _add_to_tf_record(file_annotation_path, tf_record_file, net_size):
    f = open(file_annotation_path, 'r')
    with tf.io.TFRecordWriter(tf_record_file) as writer:
        for line in tqdm.tqdm(f.readlines()):
            ann = line.strip().split(' ')
            if (len(ann) > 2):
                example = _convert_to_example(ann[0], ann[1], ann[2:], net_size)
            else:
                example = _convert_to_example(ann[0], ann[1], np.zeros(4, dtype=np.float32), net_size)
            writer.write(example)
    f.close()


if __name__ == '__main__':
    args = sys.argv[1:]
    tf_record_file = DATA_GENERATED_FOLDER + args[0] + '/tf_ann_train.tfrecords'
    file_ann = DATA_GENERATED_FOLDER + args[0] + '/annotation_file.txt'
    _add_to_tf_record(file_ann, tf_record_file, int(args[0]))

        
