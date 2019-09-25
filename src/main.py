import tensorflow as tf
tf.enable_eager_execution()
from mtcnn_arch import PNet, RNet
from dataset import MTCNNDataset
from trainer import Trainer
import numpy as np
import datetime
import argparse
import sys

def parse_arguments(argv):

    parser = argparse.ArgumentParser()

    parser.add_argument('--net_model', type=str,
                        help='Specify network model')
    
    parser.add_argument('--data_path', type=str,
                        help='Specify path to folder containing the tfrecords file')
    
    parser.add_argument('--batch_size', type=int,
                        help='Specify batch_size for network')

    parser.add_argument('--n_epoch', type=int,
                        help='Specify nb_epoch for network')
    
    return parser.parse_args(argv)



def main(args):
    network_model = {'PNET': PNet, 'RNET': RNet}

    
    net = network_model[args.net_model]()
    
    if args.net_model == 'RNET' or args.net_model == 'ONET':
        mini_batch = True 

    optimizer = tf.keras.optimizers.Adam(0.001)

    data_path = args.data_path

    dataset = MTCNNDataset(data_path)


    trainer = Trainer(net=net, train_dataset=dataset.get_training_set(batch_size=args.batch_size), 
        val_dataset=dataset.get_validation_set(batch_size=args.batch_size), optimizer=optimizer)#dataset.get_validation_set(batch_size=args.batch_size), optimizer=optimizer)
    trainer.train(n_epoch=args.n_epoch, mini_batch=mini_batch)

if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))