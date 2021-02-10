from art.utils import load_cifar10

class ImageData():
    def __init__(self, dataset_name):
        if dataset_name == 'mnist': 
            print('To-Do')

        elif dataset_name == 'cifar10': 
            (x_train, y_train), (x_val, y_val), _, _ = load_cifar10()

        if np.max(x_train) > 1: x_train = x_train.astype('float32')/255
        if np.max(x_val) > 1: x_val = x_val.astype('float32')/255
        self.clip_min = 0.0
        self.clip_max = 1.0

        self.x_train = x_train
        self.x_val = x_val
        self.y_train = y_train
        self.y_val = y_val

    def split_data(x, y, model, num_classes = 10, split_rate = 0.8, sample_per_class = 100):