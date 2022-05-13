"""
Copyright (C) 2019 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""

from __future__ import print_function, division  # division:使用//表示向下取整

import sys

sys.path.append("..")
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import time
import os
import scipy.io
import yaml
from reIDmodel import ft_net, ft_netAB, ft_net_dense, PCB, PCB_test

######################################################################
# Options
# --------
parser = argparse.ArgumentParser(description="Training")
parser.add_argument(
    "--gpu_ids", default="0", type=str, help="gpu_ids: e.g. 0  0,1,2  0,2"
)
# parser.add_argument('--which_epoch',default=90000, type=int, help='80000') # 根据实际情况修改
parser.add_argument("--which_epoch", default=100000, type=int, help="80000")  # 根据实际情况修改
# parser.add_argument('--test_dir',default='../../Market/pytorch',type=str, help='./test_data') # 需要修改
parser.add_argument(
    "--test_dir",
    default="/home/ljl/Datasets/Market1501/Market-1501-simplify/pytorch/",
    type=str,
    help="./test_data",
)  # 需要修改
# parser.add_argument('--name', default='test', type=str, help='save model path') # 需要修改 E0.5new_reid0.5_w30000
parser.add_argument(
    "--name", default="E0.5new_reid0.5_w30000", type=str, help="save model path"
)  # 需要修改 E0.5new_reid0.5_w30000
# parser.add_argument('--batchsize', default=80, type=int, help='batchsize') # 根据实际情况修改
parser.add_argument("--batchsize", default=10, type=int, help="batchsize")  # 根据实际情况修改
parser.add_argument("--use_dense", action="store_true", help="use densenet121")
parser.add_argument("--PCB", action="store_true", help="use PCB")
parser.add_argument("--multi", action="store_true", help="use multiple query")

opt = parser.parse_args()

str_ids = opt.gpu_ids.split(",")
which_epoch = opt.which_epoch
name = opt.name
test_dir = opt.test_dir

gpu_ids = []
for str_id in str_ids:
    id = int(str_id)
    if id >= 0:
        gpu_ids.append(id)

# set gpu ids
if len(gpu_ids) > 0:
    torch.cuda.set_device(gpu_ids[0])  # 指定显卡

######################################################################
# Load Data
# ---------
#
# We will use torchvision and torch.utils.data packages for loading the
# data.
#
data_transforms = transforms.Compose(
    [
        transforms.Resize((256, 128), interpolation=3),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]
)

if opt.PCB:
    data_transforms = transforms.Compose(
        [
            transforms.Resize((384, 192), interpolation=3),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )


data_dir = test_dir
image_datasets = {
    x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms)
    for x in ["gallery", "query", "multi-query"]
}
dataloaders = {
    x: torch.utils.data.DataLoader(
        image_datasets[x], batch_size=opt.batchsize, shuffle=False, num_workers=16
    )
    for x in ["gallery", "query", "multi-query"]
}

class_names = image_datasets["query"].classes
use_gpu = torch.cuda.is_available()

######################################################################
# Load model
# ---------------------------
def load_network(network):
    save_path = os.path.join(
        "../outputs", name, "checkpoints/id_%08d.pt" % opt.which_epoch
    )  # save path : ../outputs/test/checkpoints/id_num(which_epoch)
    state_dict = torch.load(save_path)
    network.load_state_dict(
        state_dict["a"], strict=False
    )  # strict=False : 不要求预训练权重层数的键值与新构建的权重层数完全吻合，只使用训练权重中与新构建网络相匹配的键值
    return network


######################################################################
# Extract feature
# ----------------------
#
# Extract feature from  a trained model.
#
def fliplr(img):
    """flip horizontal"""
    inv_idx = torch.arange(
        img.size(3) - 1, -1, -1
    ).long()  # N x C x H x W  # long为一种数据类型，该类型数据可用作索引 详见：https://blog.csdn.net/qq_29007291/article/details/91041914
    img_flip = img.index_select(
        3, inv_idx
    )  # index_select(dim,index) 从最后一列开始依次往左走，即通过图像按列倒序，实现图像水平翻转
    return img_flip


def norm(f):
    f = f.squeeze()  # 删除shape中的1维度，例shape0:(1,10,1)--shape1:(10,)
    fnorm = torch.norm(f, p=2, dim=1, keepdim=True)  # 计算范数
    f = f.div(fnorm.expand_as(f))  # expand_as : 维度扩展为同f相同的维度  div : 使用f除以f的范数，达到标准化的目的
    return f


def extract_feature(model, dataloaders):
    features = torch.FloatTensor()
    count = 0
    for data in dataloaders:
        img, label = data
        n, c, h, w = img.size()
        count += n
        if opt.use_dense:
            ff = torch.FloatTensor(n, 1024).zero_()
        else:
            ff = torch.FloatTensor(n, 1024).zero_()
        if opt.PCB:
            ff = torch.FloatTensor(n, 2048, 6).zero_()  # we have six parts
        for i in range(2):  # 随机进行水平翻转
            if i == 1:
                img = fliplr(img)
            input_img = Variable(
                img.cuda()
            )  # Variable封装Tensor并整合反向传播相关实现（tensor变成variable之后才能反向传播求梯度）
            f, x = model(input_img)  ## 疑问记录：f,x的具体含义？？
            x[0] = norm(x[0])
            x[1] = norm(x[1])
            f = torch.cat((x[0], x[1]), dim=1)  # use 512-dim feature （n,512）拼接后f的内容就是x
            f = f.data.cpu()  # (n,1024)
            ff = ff + f  # (n,1024)

        ff[:, 0:512] = norm(ff[:, 0:512])  ## 疑问记录：之前x已经做过标准化L131-L314，为什么ff要再做一次标准化？？
        ff[:, 512:1024] = norm(ff[:, 512:1024])

        # norm feature
        if opt.PCB:
            # feature size (n,2048,6)
            # 1. To treat every part equally, I calculate the norm for every 2048-dim part feature.
            # 2. To keep the cosine score==1, sqrt(6) is added to norm the whole feature (2048*6).
            fnorm = torch.norm(ff, p=2, dim=1, keepdim=True) * np.sqrt(6)
            ff = ff.div(fnorm.expand_as(ff))
            ff = ff.view(
                ff.size(0), -1
            )  # view()函数用于改变tensor的形状，-1会自适应的调整剩余维度，整体作用：把特征展平

        features = torch.cat((features, ff), 0)
    return features


def get_id(img_path):
    camera_id = []
    labels = []
    for path, v in img_path:
        # filename = path.split('/')[-1]
        filename = os.path.basename(path)
        label = filename[0:2]
        camera = filename.split("_")[1]
        if label[0:2] == "-1":
            labels.append(-1)
        else:
            labels.append(int(label))
        camera_id.append(int(camera[0]))
    return camera_id, labels


gallery_path = image_datasets["gallery"].imgs
query_path = image_datasets["query"].imgs
mquery_path = image_datasets["multi-query"].imgs

gallery_cam, gallery_label = get_id(gallery_path)
query_cam, query_label = get_id(query_path)
mquery_cam, mquery_label = get_id(mquery_path)

######################################################################
# Load Collected data Trained model
print("-------test-----------")

###load config###
config_path = os.path.join(
    "../outputs", name, "config.yaml"
)  # config_path : ../outputs/test/config.yaml config.yaml需要修改
with open(config_path, "r") as stream:
    config = yaml.load(stream, Loader=yaml.FullLoader)

model_structure = ft_netAB(
    config["ID_class"],
    norm=config["norm_id"],
    stride=config["ID_stride"],
    pool=config["pool"],
)

if opt.PCB:
    model_structure = PCB(config["ID_class"])

model = load_network(model_structure)

# Remove the final fc layer and classifier layer
model.model.fc = nn.Sequential()  # nn.sequential(要添加的神经网络模块)，如果括号中什么都没有，表明没有添加任何模块
model.classifier1.classifier = nn.Sequential()
model.classifier2.classifier = nn.Sequential()

# Change to test mode
model = model.eval()
if use_gpu:
    model = model.cuda()

# Extract feature
since = time.time()
with torch.no_grad():
    gallery_feature = extract_feature(model, dataloaders["gallery"])
    query_feature = extract_feature(model, dataloaders["query"])
    time_elapsed = time.time() - since
    print(
        "Extract features complete in {:.0f}m {:.0f}s".format(
            time_elapsed // 60, time_elapsed % 60
        )
    )
    if opt.multi:
        mquery_feature = extract_feature(model, dataloaders["multi-query"])

# Save to Matlab for check
result = {
    "gallery_f": gallery_feature.numpy(),
    "gallery_label": gallery_label,
    "gallery_cam": gallery_cam,
    "query_f": query_feature.numpy(),
    "query_label": query_label,
    "query_cam": query_cam,
}
scipy.io.savemat("pytorch_result.mat", result)
if opt.multi:
    result = {
        "mquery_f": mquery_feature.numpy(),
        "mquery_label": mquery_label,
        "mquery_cam": mquery_cam,
    }
    scipy.io.savemat("multi_query.mat", result)

os.system("python evaluate_gpu.py")
