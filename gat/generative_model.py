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

        robust_classifier = Model(mode='eval', var_scope='classifier')
        classifier_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,
                                    scope='classifier')
        classifier_saver = tf.train.Saver(var_list=classifier_vars)
        classifier_checkpoint = '../GAT/cifar10/GAT-CIFAR10/models/adv_trained_prefixed_classifier/checkpoint-70000'

        factory = BaseDetectorFactory()
        self.sess = tf.Session()
        classifier_saver.restore(self.sess, classifier_checkpoint)
        factory.restore_base_detectors(self.sess)
        base_detectors = factory.get_base_detectors()
        self.bayes_classifier = BayesClassifier(base_detectors)

        self.acc, self.th = self.initialize_threshold(accuracy)

    def initialize_threshold(self, accuracy):
        cifar = cifar10_input.CIFAR10Data('../GAT/cifar10/GAT-CIFAR10/cifar10_data')
        eval_data = cifar.eval_data
        x = eval_data.xs.astype(np.float32)
        y = eval_data.ys.astype(np.int32)

        ths = self.bayes_classifier.logit_ths
        nat_accs = self.bayes_classifier.nat_accs(x, y, self.sess)
        idx = (np.abs(np.array(nat_accs) - accuracy)).argmin()
        return nat_accs[idx], ths[idx]

    def predict(self, x):
        logits = self.bayes_classifier.batched_run(self.bayes_classifier.logits, x, self.sess)
        preds = np.argmax(logits, axis=1)
        p_x = np.max(logits, axis=1)
        preds[p_x<=self.th] = -1
        return preds
