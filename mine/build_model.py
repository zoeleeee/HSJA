from mine.dataset import encode
from mine.distance import hamming, euclidean
import torch 
import numpy as np

class ImageModel():
    def __init__(self, model_name, dataset_name, accuracy, train = False, load = False, **kwargs):
        self.model_name = model_name
        self.dataset_name = dataset_name
        self.data_model = dataset_name + model_name
        self.framework = 'pytorch'
        self.num_classes = 10
        self.metric = 'euclidean'

        print('Load network...')
        self.nb_model = 3
        model_file_form = '../cifar_update/models/cifar10/256.32_cifar10_5_{}.pt'
        self.models = []
        for i in range(self.nb_model):
            model = torch.load(model_file_form.format(i))
            model = torch.nn.DataParallel(model).cuda()
            self.models.append(model.eval())

        self.label_reps = []
        for i in range(self.nb_model):
            np.random.seed(i*5)
            self.label_reps.append(np.random.permutation(np.load('../cifar_update/data/5_label_permutation.npy')))
        self.label_reps = np.hstack(self.label_reps)

        self.th, self.acc = initialize_threshold(accuracy)

    def initialize_threshold(self, accuracy):
        scores = np.load('../cifar_update/preds/cifar10/256.32_cifar10_5_c_art_test.npy')
        y = np.load('../cifar_update/data/cifar10_art_test_label.npy')
        if model.metric == 'hamming': pred_dists, correct_idx, error_idxs = hamming(scores, 0.9, self.label_reps, y)
        elif model.metric == 'euclidean': pred_dists, correct_idx, error_idxs = euclidean(scores, .9, self.label_reps, y)
        for ith in sorted(pred_dists):
            acc = np.sum(pred_dists[correct_idx] <= ith) / len(x)
            if acc >= accuracy:
                return ith, acc

    def predict(self, x, y=0, metric='hamming'):
        assert metric in ['hamming', 'euclidean']

        if np.max(x) <= 1:
            x = (x*255).astype(np.uint8)
        if len(x.shape) == 3:
            x = np.expand_dims(x, axis=0)
        assert len(x.shape)==4, 'x shape {} error'.format(x.shape)

        scores = []
        for i in range(len(self.models)):
            xx = encode(x, i, 32, 0)
            scores.append(self.models[i](torch.Tensor(xx).cuda(), 1).detach().cpu().numpy())
        scores = np.hstack(scores)

        if metric == 'hamming': dists, _, _ = hamming(scores, 0.9, self.label_reps, y)
        elif metric == 'euclidean': dists, _, _ = euclidean(scores, 0.9, self.label_reps, y)
        if dists[0] > self.th: return -1
        else: return np.argmin(dists[0])
