# python hamming_eval.py advs/normal .5 preds/... config/...
import numpy as np
import sys
import json
from scipy.special import expit
import scipy.io as scio
#from utils import load_data
import os

def hamming(scores, t, rep):
    nat_labels = np.zeros(scores.shape).astype(np.float32)
    nat_labels[scores>=t] = 1.
    if t == 1-t:
        nat_labels[scores<t] = -1.
    else:
        nat_labels[scores<=1-t] = -1.
    rep[rep==0] = -1
    preds, preds_dist, preds_score = [], [], []

    for i in range(len(nat_labels)):
        tmp = np.repeat([nat_labels[i]], rep.shape[0], axis=0)
        dists = np.sum(np.absolute(tmp-rep), axis=-1)
        min_dist = np.min(dists)
        pred_labels = np.arange(len(dists))[dists==min_dist]
        pred_scores = [np.sum([scores[i][k] if rep[j][k]==1 else 1-scores[i][k] for k in np.arange(len(scores[i]))]) for j in pred_labels]
        pred_label = pred_labels[np.argmax(pred_scores)]
        preds.append(pred_label)
        preds_dist.append(dists[pred_label])
        preds_score.append(np.max(pred_scores))

    preds = np.array(preds)
    preds_dist = np.array(preds_dist)
#    print(preds)
    return preds_dist, preds

def euclidean(scores, t, rep):
    nat_labels = scores#np.zeros(scores.shape).astype(np.float32)
    preds, preds_dist, preds_score = [], [], []

    for i in range(len(nat_labels)):
        tmp = np.repeat([nat_labels[i]], rep.shape[0], axis=0)
        dists = np.sqrt(np.sum((tmp-rep)**2,axis=-1))
        min_dist = np.min(dists)
        pred_labels = np.arange(len(dists))[dists==min_dist]
        pred_scores = [np.sum([scores[i][k] if rep[j][k]==1 else 1-scores[i][k] for k in np.arange(len(scores[i]))]) for j in pred_labels]
        pred_label = pred_labels[np.argmax(pred_scores)]
        preds.append(pred_label)
        preds_dist.append(dists[pred_label])
        preds_score.append(np.max(pred_scores))

    preds = np.array(preds)
    preds_dist = np.array(preds_dist)
    return preds_dist, preds