import tensorflow as tf
from abc import ABC, abstractmethod
from tf_utils_func import filter_negative_samples, filter_zeros_samples


class Network(ABC):
    def __init__(self, alpha_class, beta_reg, class_th):
        self.alpha_class = alpha_class
        self.beta_reg = beta_reg
        self.class_th = class_th

    @abstractmethod
    def get_input(self):
        pass
    @abstractmethod
    def get_target_outputs(self):
        pass
    
    @abstractmethod
    def evaluate_loss(self, *args, **kwargs):
        pass

    @abstractmethod
    def compute_loss(self, target_class, target_reg, pred_class, pred_reg):
        pass
    
    @abstractmethod
    def forward(self, input, trainable=True):
        pass