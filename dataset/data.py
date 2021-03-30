# This doc is used for constructing few-shot polar-sar dataset.
# We randomly select n_base classes for base class and n_novel classed
# for novel class.(n_base+n_novel=N)
# In training process, we randomly select n*n sub-images from orgin
# m*m sar image(n<m). And then we randomly select k labeled samples for
# each base class in sub-image. And also it generate a mask for all base class samples.
# We use mirror transform to padding sar image which is first used in u-net.
# In test process, we random select k labeled samples for all class and predict all samples'
# class.
# To better statistic exp result, we select some features:
# 1、each sample num;
# 2、sub-image size;
# 3、class sample connective degree.(each class sample diistribute variance)
# and so on.
from torch.utils import data
import numpy as np
from PIL import Image


def MirrorSymmetricTransform(data, padding):
    print('padding:', padding)
    if padding <= 0:
        return data
    c, w, h = data.shape
    if min([w, h, c]) != c:
        data = data.transpose(2, 0, 1)
    c, w, h = data.shape
    assert padding < h and padding < w, "w: {}, h: {}, padding: {}".format(w, h, padding)
    res = np.zeros((c, w + 2 * padding, h + 2 * padding))
    res[:, 0:padding, 0:padding] = np.flip(data[:, 0:padding, 0:padding], axis=(1, 2))
    res[:, 0:padding, padding:-padding] = np.flip(data[:, 0:padding, :], axis=1)
    res[:, 0:padding, -padding:] = np.flip(data[:, 0:padding, h - padding:h], axis=(1, 2))
    res[:, padding:-padding, 0:padding] = np.flip(data[:, :, 0:padding], axis=2)
    res[:, padding:-padding, padding:-padding] = data
    res[:, padding:-padding, -padding:] = np.flip(data[:, :, -padding:], axis=2)
    res[:, -padding:, 0:padding] = np.flip(data[:, -padding:, 0:padding], axis=(1, 2))
    res[:, -padding:, padding:-padding] = np.flip(data[:, -padding:, :], axis=1)
    res[:, -padding:, -padding:] = np.flip(data[:, -padding:, -padding:], axis=(1, 2))
    return res;


def TestForMirrorSymmetricTransform():
    file_path = '/home/lz/few-shot-complex-polar-sar-image-classification/dataset/11.jpg'
    data = np.array(Image.open(file_path))
    print(data.shape)
    data = MirrorSymmetricTransform(data, 1000)
    img = Image.fromarray(data.astype('uint8').transpose((1, 2, 0))).convert('RGB')
    img.save('./test.jpg')


class FewShotPolarSar(data.Dataset):
    def __init__(self, data, label, padding, num_base, size, num_sample_each_class):
        # shape: C, H, W
        self.data = data
        self.c, self.h, self.w = data.shape
        # shape: 1, H, W
        self.label = label
        self.padding = padding
        # original image with symmetric mirror padding
        self.data = MirrorSymmetricTransform(self.data, padding)
        self.label_set = set(self.label.reshape(-1).astype('int32'))
        print(self.label_set)
        print("label set: ", self.label_set)
        class_label = [cls for cls in self.label_set if cls != 17]
        np.random.shuffle(class_label)
        self.base_class = class_label[:num_base]
        self.novel_class = class_label[num_base:]
        self.k = num_sample_each_class
        self.size = size

    def __len__(self):
        return int(self.h * self.w / self.size ** 2)

    def __getitem__(self, index):
        return self.getOneEpisodeData()

    def getOneEpisodeData(self):
        x, y = np.random.randint(self.size // 2, self.h - self.size // 2, 1)[0], np.random.randint(self.size // 2,
                                                                                                   self.w - self.size // 2,
                                                                                                   1)[0]
        sub_iamge = self.data[:, x - self.size // 2:x + self.size // 2 + self.padding * 2,
                    y - self.size // 2:y + self.size // 2 + self.padding * 2]
        sub_label = self.label[0, x - self.size // 2:x + self.size // 2,
                    y - self.size // 2:y + self.size // 2]
        mask = np.zeros(sub_label.shape, dtype=np.int32)
        for cls in self.base_class:
            indexs = np.argwhere(sub_label == cls)
            np.random.shuffle(indexs)
            indexs = indexs[:self.k]
            for ind in indexs:
                mask[ind[0], ind[1]] = 1
        return sub_iamge.astype('float32'), sub_label, mask


if __name__ == '__main__':
    TestForMirrorSymmetricTransform()
