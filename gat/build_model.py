import tensorflow as tf
import numpy as np
from gat.model import Model, BayesClassifier
from eval_utils import *

class ImageModel():
    def __init__(self, model_name, dataset_name, train = False, load = False, **kwargs):
        self.model_name = model_name
        self.dataset_name = dataset_name
        self.data_model = dataset_name + model_name
        self.framework = 'tensorflow'
        self.num_classes = 10

        self.th = 

    def predict(self, x):
    	classifier = Model(mode='eval', var_scope='classifier')
		classifier_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,
                                    scope='classifier')
		classifier_saver = tf.train.Saver(var_list=classifier_vars)
		classifier_checkpoint = 'models/naturally_trained_prefixed_classifier/checkpoint-70000'

		factory = BaseDetectorFactory()

    	with tf.Session() as sess:
    		classifier_saver.restore(sess, classifier_checkpoint)
    		factory.restore_base_detectors(sess)

    		base_detectors = factory.get_base_detectors()
    		bayes_classifier = BayesClassifier(base_detectors)
    		nat_preds = batched_run(classifier.predictions, classifier.x_input, x, sess)
    		det_logits = get_det_logits(x, nat_preds, detectors, sess)
    		if det_logits[0] <= self.th:
    			return -1
    		else:
    			return nat_preds[0]