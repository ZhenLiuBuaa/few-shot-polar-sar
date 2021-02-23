import numpy as np
from PIL import Image
import scipy.io as sio


class AverageNum():
    def __init__(self, num=0, sum=0):
        self.num = num
        self.sum = sum

    def update(self, num, sum):
        self.num += num
        self.sum += sum

    def __add__(self, other):
        self.num += other.num
        self.sum += other.sum
        return AverageNum(self.num + other.num, self.sum + other.sum)

    def __str__(self):
        return "Sample num: {}, Sample sum: {}, Average: {:.4f}".format(self.num, self.sum,
                                                                        self.num / (self.sum + 1e-10))

    def __repr__(self):
        return self.__str__()


def test():
    data_path = '/home/lz/few-shot-complex-polar-sar-image-classification/data/Flavoland'
    load_data_c = sio.loadmat(data_path + '_c.mat')  # dict-numpy.ndarray
    load_data_l = sio.loadmat(data_path + '_l.mat')
    load_data_p = sio.loadmat(data_path + '_p.mat')
    c12_im = load_data_c['c22']
    c12_im = Z_score(c12_im)
    print(c12_im.shape)
    data = (c12_im - c12_im.min()) / (c12_im.max() - c12_im.min()) * 255
    img = Image.fromarray(data.astype('uint8')).convert('L')
    print(img.size)
    img.save('./test22.jpg')
    print('end！！！！！！！！！！！！')

def Z_score(data):
    # Z-score nomalization
    data = (data-data.mean())/data.std()
    return data

def LoadThreeBandImage(data_path):
    load_data_c = sio.loadmat(data_path + '_c.mat')  # dict-numpy.ndarray
    load_data_l = sio.loadmat(data_path + '_l.mat')
    load_data_p = sio.loadmat(data_path + '_p.mat')
    row, col = 1079, 1024
    channel_num = 18
    data_real = np.zeros([row, col, channel_num])
    data_imag = np.zeros([row, col, channel_num])
    res = []
    for data in [load_data_c, load_data_l, load_data_p]:
        res.append(Z_score(data['c11']))
        res.append(Z_score(data['c22']))
        res.append(Z_score(data['c33']))
        res.append(Z_score(data['c12_im']))
        res.append(Z_score(data['c12_re']))
        res.append(Z_score(data['c13_im']))
        res.append(Z_score(data['c13_re']))
        res.append(Z_score(data['c23_im']))
        res.append(Z_score(data['c23_re']))
    res = [r.reshape((1, *(r.shape))) for r in res]
    res = np.concatenate(res)
    print(res.shape)
    return res


def LoadLabel(label_path):
    label = sio.loadmat(label_path)
    print(label['clas1'].shape)
    return label['clas1'].reshape((1, *(label['clas1'].shape)))


if __name__ == '__main__':
    # LoadLabel('/home/lz/few-shot-complex-polar-sar-image-classification/data/label.mat')
    test()