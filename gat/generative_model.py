import tensorflow as tf
import numpy as np
from gat.model import Model, BayesClassifier
from gat.eval_utils import *
from art.utils import load_cifar10

class ImageModel():
    def __init__(self, model_name, dataset_name, accuracy, train = False, load = False, **kwargs):
        self.model_name = model_name
        self.dataset_name = dataset_name
        self.data_model = dataset_name + model_name
        self.framework = 'tensorflow'
        self.num_classes = 10

        factory = BaseDetectorFactory()
        self.sess = tf.Session()
        factory.restore_base_detectors(self.sess)
        base_detectors = factory.get_base_detectors()
        self.bayes_classifier = BayesClassifier(base_detectors)

        (_, _), (x_test, y_test), _, _ = load_cifar10()
        self.acc, self.th = self.initialize_threshold(x_test, y_test, accuracy)

    def initialize_threshold(self, x, y, accuracy):
        ths = self.bayes_classifier.logit_ths
        nat_accs = self.bayes_classifier.nat_accs(x, y, self.sess)
        idx = (np.abs(np.array(nat_accs) - accuracy)).argmin()
        return nat_accs[idx], ths[idx]

    def predict(self, x):
        logits = self.bayes_classifier.forward(x)
        logits = self.bayes_classifier.batched_run(logits, x, self.sess)
        preds = np.argmax(logits, axis=1)
        p_x = np.max(logits, axis=1)
        preds[p_x>self.th] = -1
        return preds
