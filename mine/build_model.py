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
        self.device_ids = [0]#,1,2]

        print('Load network...')
        self.nb_model = 3
        model_file_form = '../cifar_update/models/cifar10/256.32_cifar10_5_{}.pt'
        self.models = []
        for i in range(self.nb_model):
            model = torch.load(model_file_form.format(i))
            self.models.append(model.eval())

        self.label_reps = []
        for i in range(self.nb_model):
            np.random.seed(i*5)
            self.label_reps.append(np.random.permutation(np.load('../cifar_update/data/5_label_permutation.npy')))
        self.label_reps = np.hstack(self.label_reps)
        print(self.label_reps)

        self.th, self.acc = self.initialize_threshold(accuracy)
        print('model threshold:{}, accuracy:{}'.format(self.th, self.acc))

    def initialize_threshold(self, accuracy):
        scores = np.load('../cifar_update/preds/cifar10/256.32_cifar10_5_c_cifar10_test.npy')
        y = np.load('../cifar_update/data/cifar10_test_label.npy')
        if self.metric == 'hamming': pred_dists, preds = hamming(scores, 0.9, self.label_reps)
        elif self.metric == 'euclidean': pred_dists, preds = euclidean(scores, .9, self.label_reps)
        for ith in sorted(pred_dists):
            acc = np.mean(np.logical_and(pred_dists<ith, preds==y))
            if acc >= accuracy:
                return ith, acc

    def predict(self, x, metric='hamming'):
        assert metric in ['hamming', 'euclidean']

        if np.max(x) <= 1:
            x = (x*255).astype(np.uint8)
        if len(x.shape) == 3:
            x = np.expand_dims(x, axis=0)
        assert len(x.shape)==4, 'x shape {} error'.format(x.shape)
        if np.argmin(x.shape[1:]) == 2:
            x = np.transpose(x, ((0,3,1,2)))

        scores = []
        for i in range(len(self.models)):
            model = torch.nn.DataParallel(self.models[i], self.device_ids)
            model = model.to(f'cuda:{self.device_ids[0]}')
            score = []
            for ii in range(0, len(x), 50):
                batch_x = x[ii:min(ii+50, len(x))]
                       
                xx = encode(batch_x, i, 32, 1)
                xx = torch.Tensor(xx).to(f'cuda:{self.device_ids[0]}')
                score.append(model(xx, 1).detach().cpu().numpy())
            scores.append(np.vstack(score))
        scores = np.hstack(scores)
        if metric == 'hamming': dists, preds = hamming(scores, 0.9, self.label_reps)
        elif metric == 'euclidean': dists, preds = euclidean(scores, 0.9, self.label_reps)

        preds[dists > self.th] = -1
        return preds
