import tensorflow as tf
import numpy as np
from gat.model import Model, BayesClassifier
from eval_utils import *
import gat.cifar10_input

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
        classifier_checkpoint = 'models/naturally_trained_prefixed_classifier/checkpoint-70000'

        factory = BaseDetectorFactory()

        self.sess = tf.Session()
        classifier_saver.restore(self.sess, classifier_checkpoint)
        factory.restore_base_detectors(self.sess)
        self.base_detectors = factory.get_base_detectors()
        self.acc, self.th = self.initialize_threshold(accuracy)

    def initialize_threshold(self, accuracy):
        (_, _), (x_test, y_test), _, _ = load_cifar10()
        x_test = x_test.astype(np.float32)
        nat_accs = get_nat_accs(x_test, y_test, logit_threshs, self.classifier,
                            self.base_detectors, self.sess)
        idxs = (np.abs(nat_accs-accuracy)).argmin()
        self.acc, self.th = self.initialize_threshold(x_test, y_test, accuracy)
        return nat_accs[idxs], logit_threshs[idxs]

    def predict(self, x):
        nat_preds = batched_run(classifier.predictions, classifier.x_input, x, sess)
        det_logits = get_det_logits(x, nat_preds, detectors, sess)
        nat_preds[det_logits<= self.th] = -1
        return nat_preds