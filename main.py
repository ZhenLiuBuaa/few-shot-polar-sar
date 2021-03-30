# -------------------------------------
# Project: few-shot polar sar image classification
# Date: 2021.01.18
# Author: zhen.liu
# All Rights Reserved
# -------------------------------------


from dataset.data import FewShotPolarSar
from model.model import SarModel, MetricModel
import torch.nn as nn
import torch
from torch.optim.lr_scheduler import StepLR
import numpy as np
from config.config import Config
import os
import argparse
from dataset.data_utils import LoadThreeBandImage, LoadLabel, AverageNum
from tqdm import tqdm

parser = argparse.ArgumentParser(description='Few-shot Polar Sar Image Classification')
# parse gpu
parser.add_argument('--gpu', type=str, default=3, metavar='GPU',
                    help="gpus, default:0")
parser.add_argument('--seed', type=int, default=1000, metavar='SEED',
                    help="random seed for code and data sample")
parser.add_argument('--exp_name', type=str, default='exp', metavar='EXPNAME',
                    help="experiment name")
parser.add_argument('--iters', type=int, default=0, metavar='ITERS',
                    help="iteration to restore params")

args = vars(parser.parse_args())
print(type(args))
print(args)
os.environ["CUDA_VISIBLE_DEVICES"] = str(args['gpu'])


def main():
    config = Config()
    config.update(args)
    img_data = LoadThreeBandImage(config.image_path)
    img_label = LoadLabel(config.label_path)
    dataset = FewShotPolarSar(img_data, img_label, config.padding, config.num_base_class, config.sub_img_size,
                              config.k_shot)
    sar_model = SarModel(img_data.shape[0], config.feature_dim, config.is_complex)
    metric_model = MetricModel(config.feature_dim, nn.CrossEntropyLoss)
    # Set optimizer and scheduler
    model_optim = torch.optim.Adam(list(sar_model.parameters()) + list(metric_model.parameters()), lr=config.lr)
    model_scheduler = StepLR(model_optim, step_size=config.step_size, gamma=0.5)
    print('Trianing:\n')
    data_loader = torch.utils.data.DataLoader(dataset, config.batch_size)
    # Set cuda device
    if torch.cuda.is_available():
        sar_model.cuda(0)
        metric_model.cuda(0)
    for ep in range(config.epoch):
        model_scheduler.step(ep)
        acc_dict = {}
        epoch_loss = AverageNum()
        for cls in dataset.label_set:
            acc_dict[cls] = AverageNum()
        # Set mode
        sar_model.train()
        metric_model.train()
        print(len(dataset))
        for sub_image, sub_label, mask in tqdm(data_loader):
            if torch.cuda.is_available():
                sub_image = sub_image.cuda(0)
                sub_label = sub_label.cuda(0)
                mask = mask.cuda(0)
            feature = sar_model(sub_image)
            loss, acc_dict_batch = metric_model(feature, sub_label, mask)
            for acc_dict_one in acc_dict_batch:
                acc_matrix = np.array([list(acc_dict_one[key]) for key in acc_dict_one])
                sample_sum_each_class = acc_matrix.sum(0)
                for i, key in enumerate(acc_dict_one.keys()):
                    acc_dict[key.item()].update(acc_matrix[i][i], sample_sum_each_class[i])
            epoch_loss.update(loss, 1)
            # update para
            sar_model.zero_grad()
            metric_model.zero_grad()
            loss.backward()
            model_optim.step()
        base_acc = AverageNum()
        novel_acc = AverageNum()
        for cls in dataset.base_class:
            base_acc = base_acc + acc_dict[cls]
        for cls in dataset.novel_class:
            novel_acc = novel_acc + acc_dict[cls]
        print("Base class:", base_acc, "   Novel class:", novel_acc)
        # print("Acc :", acc_dict)
        print("loss:", epoch_loss)

if __name__ == "__main__":
    main()





