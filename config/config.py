
class Config():
    # sub_img_size: Sub_iamge_size can impact sample classification performance,
    # With larger size, sub_image have more class num, and a low performance.
    # 
    def __init__(self):
        self.k_shot = 10
        self.sub_img_size = 256 
        self.padding = 2
        self.image_path = '/home/lz/few-shot-complex-polar-sar-image-classification/data/Flavoland'
        self.label_path = '/home/lz/few-shot-complex-polar-sar-image-classification/data/label.mat'
        self.num_base_class = 10
        self.is_complex = False
        self.epoch = 10000
        self.feature_dim = 128
        self.step_size = 1000
        self.lr = 0.001
        self.batch_size = 8
    def update(self, args):
        for key, value in args.items():
            self.__setattr__(key, value)
    def __str__(self):
        res = ''
        for key, value in self.__dict__.items():
            res += ('{}: {} \n'.format(key, value))
        return res