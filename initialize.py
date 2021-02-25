import numpy as np
import sys

s = sys.argv[-1]
if s == 'integrated':
    from gat.integrated_model import ImageModel 
elif s == 'mine':
    from mine.build_model import ImageModel 
elif s == 'generative':
    from gat.generative_model import ImageModel
elif s == 'robust':
    from robust.build_model import ImageModel

def clip_image(image, clip_min, clip_max):
    # Clip an image, or an image batch, with upper and lower threshold.
    return np.minimum(np.maximum(clip_min, image), clip_max) 


def decision_function(model, images):
    images = clip_image(images, 0, 1)
    y_pred = model.predict(images)
    return y_pred[0]

def main():
    res = {}
    model = ImageModel('resnet50', 'cifar10', 0.8725, train = False, load = True)
    np.random.seed(188)
    for i in range(1000000):
        print(i)
        random_noise = np.random.uniform(0, 1, (1,32,32,3)).astype(np.float32)
        y_pred = decision_function(model,random_noise)
        if y_pred != -1:
            if y_pred not in res.keys(): res[y_pred] = [list(random_noise[0])]
            else: res[y_pred].append(random_noise[0])
            np.save('pics/{}/random_noise_{}_1.npy'.format(s, y_pred), random_noise)
    for i in res.keys(): 
        print(i, len(res[i]))
        np.save('pics/{}/random_noise_{}.npy'.format(s, i), res[i])

def check(file):
    adv_file = 'pics/generative/random_{}.npy'
    for i in range(10):
        random_noise = np.load(adv_file.format(i))
        model = ImageModel('resnet50', 'cifar10', 0.8725, train = False, load = True)
        y_pred = decision_function(model,random_noise)
        print(i, y_pred)

if __name__ == '__main__':
    main()
    # check()
