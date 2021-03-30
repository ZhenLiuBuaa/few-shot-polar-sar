
class Config():
    def __init__(self):
        self.k_shot = 500
        self.sub_img_size = 64
        self.padding = 2
        self.image_path = '/home/lz/python/few-shot-complex-polar-sar-image-classification/data/Flavoland'
        self.label_path = '/home/lz/python/few-shot-complex-polar-sar-image-classification/data/label.mat'
        self.num_base_class = 10
        self.is_complex = False
        self.epoch = 10000
        self.feature_dim = 64
        self.step_size = 1000
        self.lr = 0.001
        self.batch_size = 4
    def update(self, args):
        for key, value in args.items():
            self.__setattr__(key, value)