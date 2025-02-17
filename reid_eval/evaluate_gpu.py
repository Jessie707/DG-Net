"""
Copyright (C) 2019 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""

import scipy.io
import torch
import numpy as np
import os
import matplotlib

matplotlib.use("agg")  # 让plt不显示输出图像
import matplotlib.pyplot as plt

#######################################################################
# Evaluate


def evaluate(qf, ql, qc, gf, gl, gc):
    query = qf.view(-1, 1)
    score = torch.mm(gf, query)  # torch.mm矩阵相乘 计算余弦相似度
    score = score.squeeze(1).cpu()  # squeeze(dim)将第dim维去掉（只有当第dim维对映的数值大小为1时才起作用）
    score = score.numpy()
    # predict index
    index = np.argsort(score)  # from small to large 返回的是排序后的索引值
    index = index[
        ::-1
    ]  # n:m:k的简称，从n开始（包括n），到m为止（不包括m），即[n,m),以k为间隔取值。这里表示倒序取全列表 取的是余弦相似度从大到小排序对应的索引
    # good index
    query_index = np.argwhere(gl == ql)  # 返回列表中所有不为0的元组的索引
    # same camera
    camera_index = np.argwhere(gc == qc)

    good_index = np.setdiff1d(
        query_index, camera_index, assume_unique=True
    )  # 返回query_index中有而camera_index中没有的元素 即相同id,不同camera才算有效重识别
    junk_index1 = np.argwhere(gl == -1)  # 无效索引，拍摄不完整
    junk_index2 = np.intersect1d(
        query_index, camera_index
    )  # 返回query_index和camera_index中的共有元素 属无效索引，id相同且属于同一camera，不算有效重识别
    junk_index = np.append(junk_index2, junk_index1)  # .flatten())

    CMC_tmp = compute_mAP(index, qc, good_index, junk_index)
    return CMC_tmp


def compute_mAP(index, qc, good_index, junk_index):
    ap = 0
    cmc = torch.IntTensor(len(index)).zero_()
    if good_index.size == 0:  # if empty
        cmc[0] = -1
        return ap, cmc

    # remove junk_index
    ranked_camera = gallery_cam[index]
    mask = np.in1d(
        index, junk_index, invert=True
    )  # 判断index是否在junk_index中，是则返回True,不在则返回False,加入invert=True标明对结果取反，即在列表中返回False,不在列表中返回True。这里将index中有junk_index的地方标记False,其他地方标记True
    mask2 = np.in1d(index, np.append(good_index, junk_index), invert=True)
    index = index[mask]  # 删除含有junk_index的索引
    ranked_camera = ranked_camera[mask]

    # find good_index index
    ngood = len(good_index)
    mask = np.in1d(index, good_index)  # 在good_index处标记True，其他地方标记False
    rows_good = np.argwhere(mask == True)  # 返回列表中所有mask=True的元组索引，rows_good即为good_index
    rows_good = rows_good.flatten()

    cmc[rows_good[0] :] = 1  # 设定从第一个good_index开始一直到最后，cmc的值为1。即将cmc列表中好的索引值置1。
    # cmc[rows_good[0:]] = 1 # 不用这个公式，这里不是代码错误!!!因为cmc曲线是取阶跃函数，第一次到1后，之后全取1。
    for i in range(ngood):
        d_recall = 1.0 / ngood
        precision = (i + 1) * 1.0 / (rows_good[i] + 1)
        if rows_good[i] != 0:
            old_precision = i * 1.0 / rows_good[i]
        else:
            old_precision = 1.0
        ap = ap + d_recall * (old_precision + precision) / 2

    return ap, cmc


######################################################################
result = scipy.io.loadmat("pytorch_result.mat")
query_feature = torch.FloatTensor(result["query_f"])
query_cam = result["query_cam"][0]
query_label = result["query_label"][0]
gallery_feature = torch.FloatTensor(result["gallery_f"])
gallery_cam = result["gallery_cam"][0]
gallery_label = result["gallery_label"][0]

multi = os.path.isfile("multi_query.mat")

if multi:
    m_result = scipy.io.loadmat("multi_query.mat")
    mquery_feature = torch.FloatTensor(m_result["mquery_f"])
    mquery_cam = m_result["mquery_cam"][0]
    mquery_label = m_result["mquery_label"][0]
    mquery_feature = mquery_feature.cuda()

query_feature = query_feature.cuda()
gallery_feature = gallery_feature.cuda()

print(query_feature.shape)
alpha = [0, 0.5, -1]
for j in range(len(alpha)):
    CMC = torch.IntTensor(len(gallery_label)).zero_()
    ap = 0.0
    for i in range(len(query_label)):
        qf = query_feature[i].clone()
        if alpha[j] == -1:
            qf[0:512] *= 0
        else:
            qf[512:1024] *= alpha[j]
        ap_tmp, CMC_tmp = evaluate(
            qf,
            query_label[i],
            query_cam[i],
            gallery_feature,
            gallery_label,
            gallery_cam,
        )
        if CMC_tmp[0] == -1:
            continue
        CMC = CMC + CMC_tmp
        ap += ap_tmp

    CMC = CMC.float()
    CMC = CMC / len(query_label)  # average CMC
    print(
        "Alpha:%.2f Rank@1:%.4f Rank@5:%.4f Rank@10:%.4f mAP:%.4f"
        % (alpha[j], CMC[0], CMC[4], CMC[9], ap / len(query_label))
    )

# multiple-query
CMC = torch.IntTensor(len(gallery_label)).zero_()
ap = 0.0
if multi:
    malpha = 0.5  ######
    for i in range(len(query_label)):
        mquery_index1 = np.argwhere(mquery_label == query_label[i])
        mquery_index2 = np.argwhere(mquery_cam == query_cam[i])
        mquery_index = np.intersect1d(mquery_index1, mquery_index2)
        mq = torch.mean(mquery_feature[mquery_index, :], dim=0)
        mq[512:1024] *= malpha
        ap_tmp, CMC_tmp = evaluate(
            mq,
            query_label[i],
            query_cam[i],
            gallery_feature,
            gallery_label,
            gallery_cam,
        )
        if CMC_tmp[0] == -1:
            continue
        CMC = CMC + CMC_tmp
        ap += ap_tmp
    CMC = CMC.float()
    CMC = CMC / len(query_label)  # average CMC
    print(
        "multi Rank@1:%.4f Rank@5:%.4f Rank@10:%.4f mAP:%.4f"
        % (CMC[0], CMC[4], CMC[9], ap / len(query_label))
    )
