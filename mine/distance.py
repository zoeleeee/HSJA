# python hamming_eval.py advs/normal .5 preds/... config/...
import numpy as np
import sys
import json
from scipy.special import expit
import scipy.io as scio
#from utils import load_data
import os

def hamming(scores, t, rep, labels):
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
    correct_idxs = np.arange(len(preds))[preds == labels]
    error_idxs = np.arange(len(preds))[preds != labels]
    return preds_dist, correct_idxs, error_idxs

def hamming_idxs(scores, config, t, dataset, label_path='data/cifar10_test_label.npy'):
    #nb_labels = config['num_labels']
    st = config['seed']*config['label_length']
    if config['label_length'] == 5:
        rep = []
        for i in range(int(len(scores)/config['label_length'])):
            np.random.seed(i)
            rep.append(np.random.permutation(np.load('data/5_label_permutation.npy')))
        rep = np.hstack(rep)
    elif dataset != 'cifar100':
        rep = np.load('data/2_label_permutation.npy')[st:st+scores.shape[-1]].T
    else:
        rep = np.load('data/cifar100_2_label_permutation.npy')[st:st+scores.shape[-1]].T

    
    if label_path.find('svhn_test_label') != -1:
        test_dict = scio.loadmat('data/test_32x32.mat')
        labels = test_dict['y'].reshape(-1)
        labels[labels==10]=0
    else:
        labels = np.load(label_path)
    
    #if os.path.exists('data/{}_random_chosen_idxs.npy'.format(dataset)):
    #    idxs = np.load('data/{}_random_chosen_idxs.npy'.format(dataset))
    #    if label_path.find('TAE') != -1 and len(scores)>len(idxs):
    #        scores_idxs = np.load(label_path[:-9]+'idxs.npy', allow_pickle=True).reshape(-1)
    #        idxs = np.arange(len(scores_idxs))[np.array([True if i in idxs else False for i in scores_idxs])]
    #    if len(scores)>len(idxs): scores = scores[idxs]
    #    labels = labels[idxs]
    #print(scores.shape)
   
    return hamming(scores, t, rep, labels)

def euclidean(scores, t, rep, labels):
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
#    print(preds)
    correct_idxs = np.arange(len(preds))[preds == labels]
    error_idxs = np.arange(len(preds))[preds != labels]
    return preds_dist, correct_idxs, error_idxs

def euclidean_idxs(scores, config, t, dataset, label_path='data/cifar10_test_label.npy'):
    
    #nb_labels = config['num_labels']
    st = config['seed']*config['label_length']
    if config['label_length'] == 5:
        rep = []
        for i in range(int(len(scores)/config['label_length'])):
            np.random.seed(i)
            rep.append(np.random.permutation(np.load('data/5_label_permutation.npy')))
        rep = np.hstack(rep)
    elif dataset != 'cifar100':
        rep = np.load('data/2_label_permutation.npy')[st:st+scores.shape[-1]].T
    else:
        rep = np.load('data/cifar100_2_label_permutation.npy')[st:st+scores.shape[-1]].T

    if label_path.find('svhn_test_label') != -1:
        test_dict = scio.loadmat('data/test_32x32.mat')
        labels = test_dict['y'].reshape(-1)
        labels[labels==10]=0
    else:
        labels = np.load(label_path)
    
    if os.path.exists('data/{}_random_chosen_idxs.npy'.format(dataset)):
        idxs = np.load('data/{}_random_chosen_idxs.npy'.format(dataset))
        if label_path.find('TAE') != -1 and len(scores)>len(idxs):
            scores_idxs = np.load(label_path[:-9]+'idxs.npy', allow_pickle=True).reshape(-1)
            idxs = np.arange(len(scores_idxs))[np.array([True if i in idxs else False for i in scores_idxs])]
        if len(scores)>len(idxs): scores = scores[idxs]
        labels = labels[idxs]
    #print(scores.shape)
    return euclidean(scores, st, t, dataset, labels)

if __name__ == '__main__':
    with open(sys.argv[-1]) as config_file:
      config = json.load(config_file)
    st = config['seed']*config['label_length']

    name = sys.argv[-2]
    _type = sys.argv[-4]
    t = eval(sys.argv[-3])
    dataset = sys.argv[-5]
    #model_dir = config['model_dir']
    scores = np.load(name)
#   if np.max(scores) > 1:
#       scores = expit(scores)
    # labels = np.load('preds/labels_{}'.format(name))
    print(scores.shape)
    preds_dist, correct_idxs, error_idxs = hamming_idxs(scores, config, t, dataset, _type)
    print(preds_dist.shape)
    print('avg Hamming distance:{}, max:{}, min:{}, med:{}'.format(np.mean(preds_dist), np.max(preds_dist), np.min(preds_dist), np.median(preds_dist)))

    ts = np.arange(np.max(preds_dist)+1)
#    if _type == 'advs':
#        y_true = np.ones(preds_dist.shape)
#    else:
#        y_true = np.zeros(preds_dist.shape)
    for t in ts:
#        det_advs = np.zeros(preds_dist.shape)
#        det_advs[preds_dist>t] =  1.
#        auc = roc_auc_score(y_true, det_advs)
#        print('tpr:', np.mean(det_advs), 'fnr:',) 
        print(t, 'acc:', np.sum(preds_dist[correct_idxs] < t+1)*1.0 / len(preds_dist))
        print(t, 'err:', np.sum(preds_dist[error_idxs] < t+1)*1.0 / len(preds_dist))




