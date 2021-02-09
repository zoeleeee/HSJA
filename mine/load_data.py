import numpy as np

class ImageData():
	def __init__(self, dataset_name):
		if dataset_name == 'mnist': 
			

		elif dataset_name == 'cifar10': 
			x_train = np.load('../cifar_update/data/cifar10_train_data.npy')
			y_train = np.load('../cifar_update/data/cifar10_train_label.npy')
			x_test = np.load('../cifar_update/data/cifar10_test_data.npy')
			y_test = np.load('../cifar_update/data/cifar10_test_label.npy')

		self.x_train = x_train
		self.x_val = x_test
		self.y_train = y_train
		self.y_val = y_test

def split_data(x, y, model, num_classes = 10, split_rate = 0.8, sample_per_class = 100):
	np.random.seed(10086)
	label_pred = model.predict(x)
	correct_idx = label_pred==y
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
