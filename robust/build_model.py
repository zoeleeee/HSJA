import tensorflow as tf
import numpy as np
from gat.model import Model, BayesClassifier
from gat.eval_utils import *
import gat.cifar10_input as cifar10_input

class ImageModel():
    def __init__(self, model_name, dataset_name, accuracy, train = False, load = False, **kwargs):
        self.model_name = model_name
        self.dataset_name = dataset_name
        self.data_model = dataset_name + model_name
        self.framework = 'tensorflow'
        self.num_classes = 10

        self.classifier = Model(mode='eval', var_scope='classifier')
        classifier_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,
                                    scope='classifier')
        classifier_saver = tf.train.Saver(var_list=classifier_vars)
        classifier_checkpoint = 'models/adv_trained_prefixed_classifier/checkpoint-70000'

        self.sess = tf.Session()
        classifier_saver.restore(self.sess, classifier_checkpoint)
        factory.restore_base_detectors(self.sess)
        base_detectors = factory.get_base_detectors()
        self.bayes_classifier = BayesClassifier(base_detectors)

    def predict(self, x):
         preds = batched_run(self.classifier.predictions,
                        self.classifier.x_input, x, sess)
        return preds
