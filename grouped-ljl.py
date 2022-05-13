"""
author:Jessie
data:2022-05-11
"""

import os
from shutil import copyfile

# You only need to change this line to your img path
download_path = "/home/ljl/Data/REID/ORS_REID/all/group2/"

if not os.path.isdir(download_path):
    print("please change the download_path")

save_path = download_path + "/grouped"
if not os.path.isdir(save_path):
    os.mkdir(save_path)
# -----------------------------------------
# move to dir
img_path = download_path
img_save_path = download_path + "/grouped"
if not os.path.isdir(img_save_path):
    os.mkdir(img_save_path)

for root, dirs, files in os.walk(img_path, topdown=True):
    for name in files:
        if not name[-3:] == "png":
            continue
        ID = name.split("_")
        src_path = img_path + "/" + name
        dst_path = img_save_path + "/" + ID[0]
        if not os.path.isdir(dst_path):
            os.mkdir(dst_path)
        copyfile(src_path, dst_path + "/" + name)
