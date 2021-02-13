import tensorflow as tf
import numpy as np
from gat.model import Model, BayesClassifier
from gat.eval_utils import *
from art.utils import load_cifar10
# logit_threshs = np.linspace(-300., 30.0, 1000)

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
        classifier_checkpoint = '../GAT/cifar10/GAT-CIFAR10/models/naturally_trained_prefixed_classifier/checkpoint-70000'

        factory = BaseDetectorFactory()

        self.sess = tf.Session()
        classifier_saver.restore(self.sess, classifier_checkpoint)
        factory.restore_base_detectors(self.sess)
        self.base_detectors = factory.get_base_detectors()
        self.acc, self.th = self.initialize_threshold(accuracy)

    def initialize_threshold(self, accuracy):
        cifar = cifar10_input.CIFAR10Data('../GAT/cifar10/GAT-CIFAR10/cifar10_data')
        eval_data = cifar.eval_data
        x_test = eval_data.xs.astype(np.float32)
        y_test = eval_data.ys.astype(np.int32)
        nat_accs = get_nat_accs(x_test, y_test, logit_threshs, self.classifier,
                            self.base_detectors, self.sess)
        idxs = (np.abs(np.array(nat_accs)-accuracy)).argmin()
        return nat_accs[idxs], logit_threshs[idxs]

    def predict(self, x):
        nat_preds = batched_run(self.classifier.predictions, self.classifier.x_input, x, self.sess)
        det_logits = get_det_logits(x, nat_preds, self.base_detectors, self.sess)
        nat_preds[det_logits<= self.th] = -1
        return nat_preds
