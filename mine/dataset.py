import numpy as np
from torch.utils.data import TensorDataset, DataLoader, Dataset
import math
import hashlib
import torch
import scipy.io as scio
from torchvision import transforms
import os
from art.utils import load_cifar10

#transforms = lambda size: torch.nn.Sequential(
#        transforms.RandomCrop(size, padding=4),
#        transforms.RandomHorizontalFlip(),
#        transforms.ColorJitter(.25,.25,.25),
#        transforms.RandomRotation(2),
#        transforms.ToTensor(),
#    )
#TRAIN_TRANSFORMS_DEFAULT = torch.jit.script(transforms)
#transforms = lambda size: torch.nn.Sequential(
#        transforms.Resize(size),
#        transforms.CenterCrop(size),
#        transforms.ToTensor()
#    )
#TEST_TRANSFORMS_DEFAULT = torch.jit.script(transforms)

TRAIN_TRANSFORMS_DEFAULT = lambda size: transforms.Compose([
             transforms.RandomCrop(size, padding=4),
             transforms.RandomHorizontalFlip(),
#             transforms.ColorJitter(.25,.25,.25),
             transforms.RandomRotation(2),
#             transforms.ToTensor(),
         ])

TEST_TRANSFORMS_DEFAULT = lambda size:transforms.Compose([
         transforms.Resize(size),
         transforms.CenterCrop(size),
#         transforms.ToTensor()
     ])

def rand_hash(data, seed, p):
    probs = torch.rand(len(data))
    if torch.max(data) <= 1: data *= 255
#    assert torch.max(data) == 255, "data val should be [0,255] integers, instead of [{},{}]".format(torch.min(data), torch.max(data))
    data = data.transpose(1,2).transpose(2,3).tolist()
    assert len(data.shape) == 4, "data shape should be 4, instead of {}".format(len(data.shape), data.shape)
    for i in range(len(data)):
        if probs[i] > p:
            continue
        for j in range(len(data[0])):
            for t in range(len(data[0][0])):
                for k in range(len(data[0][0][0])):
                    b = m(bytearray([data[i][j][t][k], seed])).hexdigest()
                    data[i][j][k][t] = [int(b[t:t+2], 16) for t in range(0,length*2,2)]
    return torch.Tensor(data).reshape((len(data), 32, 32, 96)).transpose(3,2).transpose(2,1)

def rand_perm(data, permutation, p):
    assert len(data.shape) == 4, "data shape should be 4, instead of {}".format(len(data.shape), data.shape)
    if torch.max(data) <= 1:
        data *= 255.
#    assert torch.max(data) == 255, "data val should be [0,255] integers, instead of [{},{}]".format(torch.min(data), torch.max(data))
    
    probs = torch.rand(len(data))
    for i in range(data.shape[0]):
        if probs[i] > p:
            continue
        for j in range(data.shape[1]):
            for t in range(data.shape[2]):
                for k in range(data.shape[3]):
                    data[i,j,k,t] = permutation[int(data[i,j,t,k])]
    return data

class CustomTensorDataset(Dataset):
    """Dataset wrapping tensors.

    Each sample will be retrieved by indexing tensors along the first dimension.
    Arguments:
         *tensors (Tensor): tensors that have the same size of the first dimension.
     """

    def __init__(self, *tensors, transform=None):
        assert len(tensors[0]) == len(tensors[1])
        self.tensors = tensors
        self.transform = transform
    def __getitem__(self, index):
        im, targ = self.tensors[0][index], self.tensors[1][index]
        im = self.transform(im)
        # transfer to PILImage, then the channal only cut 3, required under torch v1.7.0
        #if self.transform:
        #    real_transform = transforms.Compose([
        #        transforms.ToPILImage(),
        #        self.transform
        #    ])
        #    im = real_transform(im)
        return im, targ
    def __len__(self):
        return len(self.tensors[0])


def encode_channal(data, seed, length, input_bytes, channal_idx):
    slide_step = int(math.sqrt(input_bytes))
    new_data = []
    if length == 32:
        m = hashlib.sha256
    elif length == 48:
        m = hashlib.sha384
    elif length == 64:
        m = hashlib.sha512
    for a in data:
        img = []
        for i in range(0, a.shape[1]+1-slide_step):
            tmp = []
            for j in range(0, a.shape[2]+1-slide_step):
                meg = [a[channal_idx][i+t][j+k] for t in range(slide_step) for k in range(slide_step)]
                b = m(bytearray(meg+[seed])).hexdigest()[:length*2]
                tmp.append([int(b[t:t+2], 16) for t in range(0,length*2,2)])
            img.append(tmp)
        new_data.append(img)
    return np.array(new_data).astype(np.float32).transpose((0,3,1,2)) / 255.

def encode(data, seed, length, input_bytes):
    data = data.astype(np.uint8)
    assert seed<256, "invalid seed: seed must < 256"

    nb_channal = np.min(data.shape[1:])
    
    output = []
    for i in range(nb_channal):
        output.append(encode_channal(data, seed, length, input_bytes, i))
    return np.hstack(output)

def multi_label(dataset, label, start_pos, length=20):
    start_pos *= length
    np.random.seed(start_pos)
    if dataset =='cifar100': label_maps = np.load('data/cifar100_2_label_permutation.npy')[start_pos:start_pos+length].T
    elif dataset =='gtsrb': label_maps = np.load('data/gtsrb_2_label_permutation.npy')[start_pos:start_pos+length].T
    elif length == 5: label_maps = np.random.permutation(np.load('data/5_label_permutation.npy'))
    else: label_maps = np.load('data/2_label_permutation.npy')[start_pos:start_pos+length].T
    labs = [label_maps[int(i)] for i in label]
    return np.array(labs)

def make_loader(x,  y, batch_size, shuffle, sign, transform=None):
#    dataset = TensorDataset(torch.Tensor(x), torch.LongTensor(y) if sign==0 else torch.FloatTensor(y))
    print(x.shape, y.shape)
    if transform is None: 
        dataset = TensorDataset(torch.Tensor(x), torch.LongTensor(y) if sign==0 else torch.FloatTensor(y))
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=8)
    else: 
        dataset = CustomTensorDataset(torch.Tensor(x), torch.LongTensor(y) if sign==0 else torch.FloatTensor(y), transform=transform)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)#, num_workers=8)
    return loader

def get_loader(sign, batch_size, seed, len_label, length, input_bytes, dataset, path=None, _hash=True):
    assert dataset in ['cifar10', 'cifar100', 'svhn', 'gtsrb'], "dataset must in [cifar10, cifar100, 'svhn', 'gtsrb']"
    if sign == 'advs':
        data = np.load(path)
        if os.path.exists(path[:-9]+'_label.npy'):
            label = np.load(path[:-9]+'_label.npy')
        else:
            print(path + ' does not have label.npy')            
            assert os.path.exists('data/{}_random_chosen_idxs.npy'.format(dataset)), '{} does not have random chosen idxs'.format(dataset)
            if dataset == 'cifar10':
                label = np.load('data/cifar10_test_label.npy')
            elif dataset == 'svhn':
                test_dict = scio.loadmat('data/test_32x32.mat')
                label = test_dict['y'].reshape(-1)
                label[label==10] = 0
            if path.find('AEs') != -1:               
                label = np.load('data/cifar10_art_test_label.npy')
                idxs = np.load(path[:-4]+'_idxs.npy')
            else:
                idxs = np.load('data/{}_random_chosen_idxs.npy'.format(dataset))
            label = label[idxs]
    else:
        if dataset == 'cifar10':
            x_test = np.load('data/cifar10_art_test_data.npy')
            data = (x_test *255).astype(np.uint8)
            label = np.load('data/cifar10_art_test_label.npy')
            #data = np.load('data/cifar10_test_data.npy')
            #label = np.load('data/cifar10_test_label.npy')
        elif dataset == 'cifar100':
            data = np.load('data/cifar100_test_data.npy')
            label = np.load('data/cifar100_test_label.npy')
        elif dataset == 'svhn':
            test_dict = scio.loadmat('data/test_32x32.mat')
            data = test_dict['X'].transpose((3,2,0,1)).astype(np.float32)
            label = test_dict['y'].reshape(-1)
            label[label==10] = 0
        elif dataset == 'gtsrb':
            data = np.load('data/gtsrb_test_data.npy')
            label = np.load('data/gtsrb_test_labels.npy')
#    data, label = data[:10], label[:10]
    if np.argmin(data.shape) == 3:
        data = data.transpose((0,3,1,2))
    if _hash:
        test_y = multi_label(dataset, label, seed, len_label)
        test_x = encode(data, seed, length, input_bytes)
        test_loader = make_loader(test_x, test_y, batch_size, False, 1, transform=TEST_TRANSFORMS_DEFAULT(32))
    else:
        test_x = np.repeat(data, length, axis=1)
        test_loader = make_loader(test_x.astype(np.float32)/255., label, batch_size, False, 0, transform=TEST_TRANSFORMS_DEFAULT(32))

    if sign == 'train':
        if dataset == 'cifar10':
            train_x = np.load('data/cifar10_train_data.npy').astype(np.float32)
            train_y = np.load('data/cifar10_train_label.npy')
        elif dataset == 'cifar100':
            train_x = np.load('data/cifar100_train_data.npy').astype(np.float32)
            train_y = np.load('data/cifar100_train_label.npy')
        elif dataset == 'svhn':
            train_dict = scio.loadmat('data/train_32x32.mat')
            train_x = train_dict['X'].transpose((3,2,0,1)).astype(np.float32)#[:10]
            train_y = train_dict['y'].reshape(-1)#[:10]
            train_y[train_y==10] = 0
        elif dataset == 'gtsrb':
            train_x = np.load('data/gtsrb_train_data.npy')
            train_y = np.load('data/gtsrb_train_labels.npy')
        if np.argmin(train_x.shape) == 3:
            train_x = train_x.transpose((0,3,1,2))
#        train_x, train_y = train_x[:10], train_y[:10]

        if _hash:
            encode_train_x = encode(train_x, seed, length, input_bytes)
            encode_train_y = multi_label(dataset, train_y, seed, len_label)
            encode_train_loader = make_loader(encode_train_x, encode_train_y, batch_size, True, 1, transform=TRAIN_TRANSFORMS_DEFAULT(32))

        train_x = np.repeat(train_x, length, axis=1)
        data = np.repeat(data, length, axis=1)

        normal_train_loader = make_loader(train_x/255., train_y, batch_size, True, 0, transform=TRAIN_TRANSFORMS_DEFAULT(32))
        normal_val_loader = make_loader(data.astype(np.float32)/255., label, batch_size, True, 0, transform=TEST_TRANSFORMS_DEFAULT(32))

        if _hash: return encode_train_loader, test_loader, normal_train_loader, normal_val_loader
        else: return normal_train_loader, normal_val_loader
    return test_loader
