from PIL import Image
import tensorflow as tf
import numpy as np
from tqdm import tqdm
from utils.path import getfiles


class Anime(object):
    def __init__(self, image_folder=None, npy='faces/faces.npy'):
        try:
            self.data = np.load(npy)
        except:
            self.data = self._get_data(image_folder)
            self.save_txt(npy)

    def _get_data(self, image_folder):

        image_names = getfiles(image_folder)

        images = []

        for image_name in tqdm(image_names):
            image = np.array(Image.open('{0}/{1}'.format(image_folder, image_name)))
            if np.ndim(image) == 3:
                image = (image / 255.) * 2 - 1
                image = np.pad(image, ((2, 2), (2, 2), (0, 0)), mode='constant')
                images.append(image.flatten())

        images = np.array(images)

        return images

    def save_txt(self, path='faces/faces.npy'):
        np.save(path, self.data)

    def get_size(self):
        return self.data.shape[0]

    def next_batch(self, num):
        '''
        Return a total of `num` random samples and labels.
        '''
        idx = np.arange(0, self.get_size())
        np.random.shuffle(idx)
        idx = idx[:num]
        batch = self.data[idx]

        return batch