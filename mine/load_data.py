import numpy as np
from mine.distance import hamming, euclidean

class ImageData():
    def __init__(self, dataset_name):
        if dataset_name == 'mnist': 
            print('To-Do')

        elif dataset_name == 'cifar10': 
            x_train = np.load('../cifar_update/data/cifar10_train_data.npy')
            y_train = np.load('../cifar_update/data/cifar10_train_label.npy')
            x_val = np.load('../cifar_update/data/cifar10_art_test_data.npy')
            y_val = np.load('../cifar_update/data/cifar10_art_test_label.npy')

        if np.max(x_train) > 1: x_train = x_train.astype('float32')/255
        if np.max(x_val) > 1: x_val = x_val.astype('float32')/255
        self.clip_min = 0.0
        self.clip_max = 1.0

        self.x_train = x_train
        self.x_val = x_val
        self.y_train = y_train
        self.y_val = y_val

def split_data(x, y, model, num_classes = 10, split_rate = 0.8, sample_per_class = 100):
    np.random.seed(10086)
    scores = np.load('../cifar_update/preds/cifar10/256.32_cifar10_5_c_art_test.npy')
    if model.metric == 'hamming': pred_dists, correct_idx, error_idxs = hamming(scores, 0.9, model.label_reps, y)
    elif model.metric == 'euclidean': pred_dists, correct_idx, error_idxs = euclidean(scores, .9, mdoel.label_reps, y)
    correct_idx = correct_idx[pred_dists[correct_idx] <= model.th]
    print('Accuracy is {}'.format(np.mean(correct_idx)))
    x, y = x[correct_idx], y[correct_idx]
    label_pred = label_pred[correct_idx]

    x_train, x_test, y_train, y_test = [], [], [], []
    for class_id in range(num_classes):
        _x = x[label_pred == class_id][:sample_per_class]
        _y = y[label_pred == class_id][:sample_per_class]
        l = len(_x)
        x_train.append(_x[:int(l * split_rate)])
        x_test.append(_x[int(l * split_rate):])

        y_train.append(_y[:int(l * split_rate)])
        y_test.append(_y[int(l * split_rate):])

    x_train = np.concatenate(x_train, axis = 0)
    x_test = np.concatenate(x_test, axis = 0)
    y_train = np.concatenate(y_train, axis = 0)
    y_test = np.concatenate(y_test, axis = 0)

    idx_train = np.random.permutation(len(x_train))
    idx_test = np.random.permutation(len(x_test))

    x_train = x_train[idx_train]
    y_train = y_train[idx_train]

    x_test = x_test[idx_test]
    y_test = y_test[idx_test]

    return x_train, y_train, x_test, y_test
