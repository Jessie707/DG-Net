"""
author:Jessie
data:2022-05-12
"""
import os
from re import I
import shutil

img_root_path = "/home/ljl/Data/REID/ORS_REID"

if not os.path.isdir(img_root_path):
    print("please change the root_path")

img_grouped_path = img_root_path + "/all/merge"

# --------------------------------
# 以单双号划分训练，测试集
# --------------------------------
# gallery（odd）
gallery_path = img_root_path + "/gallery"
if not os.path.isdir(gallery_path):
    os.mkdir(gallery_path)
target_root_path = gallery_path
for root, dirs, files in os.walk(img_grouped_path, topdown=True):
    for dir in dirs:
        if (int(dir) % 2) != 0:
            src_path = os.path.join(root, dir)
            tar_path = target_root_path + "/" + dir
            if not os.path.isdir(tar_path):
                shutil.copytree(src_path, tar_path)

# --------------------------------
# train_all（even）
trainall_path = img_root_path + "/train_all"
if not os.path.isdir(trainall_path):
    os.mkdir(trainall_path)
target_root_path = trainall_path
for root, dirs, files in os.walk(img_grouped_path, topdown=True):
    for dir in dirs:
        if (int(dir) % 2) == 0:
            src_path = os.path.join(root, dir)
            tar_path = target_root_path + "/" + dir
            if not os.path.isdir(tar_path):
                shutil.copytree(src_path, tar_path)

# --------------------------------
# 从gallery中分出query
query_path = img_root_path + "/query"
if not os.path.isdir(query_path):
    os.mkdir(query_path)
for root, dirs, files in os.walk(gallery_path, topdown=True):
    for dir in dirs:
        for rootin, dirsin, filesin in os.walk(
            os.path.join(gallery_path, dir), topdown=True
        ):
            i = 0
            for name in filesin:
                if not name[-3:] == "png":
                    continue
                if i == 0:
                    ID = name.split("_")
                    src = os.path.join(rootin, name)
                    tar = query_path + "/" + ID[0]
                    if os.path.isdir(tar):
                        shutil.rmtree(tar)
                    if not os.path.isdir(tar):
                        os.mkdir(tar)
                    shutil.copyfile(src, tar + "/" + name)
                    i += 1

# --------------------------------
# 从train_all中分出train,val
train_path = img_root_path + "/train"
if not os.path.isdir(train_path):
    os.mkdir(train_path)
val_path = img_root_path + "/val"
if not os.path.isdir(val_path):
    os.mkdir(val_path)
for root, dirs, files in os.walk(trainall_path, topdown=True):
    for dir in dirs:
        for rootin, dirsin, filesin in os.walk(
            os.path.join(trainall_path, dir), topdown=True
        ):
            i = 0
            for name in filesin:
                if not name[-3:] == "png":
                    continue
                if i == 0:
                    ID = name.split("_")
                    src = os.path.join(rootin, name)
                    tar = val_path + "/" + ID[0]
                    if os.path.isdir(tar):
                        shutil.rmtree(tar)
                    if not os.path.isdir(tar):
                        os.mkdir(tar)
                    shutil.copyfile(src, tar + "/" + name)
                    i += 1
                else:
                    ID = name.split("_")
                    src = os.path.join(rootin, name)
                    tar = train_path + "/" + ID[0]
                    if os.path.isdir(tar):
                        shutil.rmtree(tar)
                    if not os.path.isdir(tar):
                        os.mkdir(tar)
                    shutil.copyfile(src, tar + "/" + name)
                    i += 1
