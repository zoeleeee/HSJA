import numpy as np
import sys

s = sys.argv[-1]
if s == 'integrated':
	from gat.integrated_model import ImageModel 
elif s == 'mine':
	from mine.build_model import ImageModel 

def clip_image(image, clip_min, clip_max):
    # Clip an image, or an image batch, with upper and lower threshold.
    return np.minimum(np.maximum(clip_min, image), clip_max) 


def decision_function(model, images, target_label):
    images = clip_image(images, 0, 1)
    y_pred = model.predict(images)
    y_pred[y_pred==-1] = original_label
    return y_pred == target_label

def main():
	model = ImageModel('resnet50', 'cifar10', 0.8725, train = False, load = True)
	for targ in range(10):
		while True:
		    random_noise = np.random.uniform(0, 1, (1,32,32,3))
		    success = decision_function(model,random_noise, targ)
		    if success:
		        break
		np.save('pics/{}/random_noise_{}.npy'.format(s, targ), random_noise)

if __name__ == '__main__':
	main()