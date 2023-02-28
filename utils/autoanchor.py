# YOLOv5 🚀 by Ultralytics, GPL-3.0 license
"""
AutoAnchor utils
"""

import random

import numpy as np
import torch
import yaml
from tqdm import tqdm

from utils import TryExcept
from utils.general import LOGGER, colorstr


PREFIX = colorstr('AutoAnchor: ')


def check_anchor_order(m):
    """用在check_anchors函数
    确认anchors和stride的顺序是一致的
    Check anchor order against stride order for YOLOv5 Detect() module m, and correct if necessary
    Arguments:
        m: model中的最后一层 Detect层
    """
    # 计算anchor的面积 anchor area [9]
    a = m.anchors.prod(-1).mean(-1).view(-1)  # mean anchor area per output layer
    # 计算最大anchor与最小anchor面积差
    da = a[-1] - a[0]  # delta a
    # 计算最大stride与最小stride差
    ds = m.stride[-1] - m.stride[0]  # delta s
    # torch.sign(x):当x大于/小于0时，返回1/-1
    # 如果这里anchor与stride顺序不一致，则重新调整顺序
    if da and (da.sign() != ds.sign()):  # same order
        LOGGER.info(f'{PREFIX}Reversing anchor order')
        m.anchors[:] = m.anchors.flip(0)


@TryExcept(f'{PREFIX}ERROR')
def check_anchors(dataset, model, thr=4.0, imgsz=640):
    """用于train.py中，通过bpr确定是否需要改变anchors，需要就调用K-means重新计算anchors
    Check anchor fit to data, recompute if necessary
    Arguments:
        dataset: 自定义数据集
        model: 初始化的模型
        thr: 超参中得到 界定anchor与label匹配程度的阈值
        imgsz: 图片尺寸 默认640
    """
    # m: 从model中取出最后一层(Detect)
    m = model.module.model[-1] if hasattr(model, 'module') else model.model[-1]  # Detect()
    # dataset.shapes.max(1, keepdims=True): 每张图片的较长边
    # dataset.shapes: {ndarray:(6160,2)}
    # shapes: 将数据集图片的最长边缩放到img_size，较小边相应缩放 得到新的所有数据集图片的宽高
    shapes = imgsz * dataset.shapes / dataset.shapes.max(1, keepdims=True)
    # 产生随机数scale: {ndarray:(6160,1)}
    scale = np.random.uniform(0.9, 1.1, size=(shapes.shape[0], 1))  # augment scale
    # 所有target的wh，基于原图大小 shape * scale: 随机化尺度变化 wh: {Tensor:(6071,2)}
    wh = torch.tensor(np.concatenate([l[:, 3:5] * s for s, l in zip(shapes * scale, dataset.labels)])).float()  # wh

    def metric(k):  # compute metric
        """用在check_anchors函数中 计算性能
        根据数据集的所有图片的wh和当前所有anchors k计算bpr(best possible recall) 和 aat(anchors above threshold)
        Arguments:
            k: anchors [9, 2] wh: [6071, 2]

        Return:
            bpr: 最多能被找回(通过thr)de gt框数量 / 所有gt框数量    小于0.98才会用k-means计算anchors
            aat: 每个target平均有多少个anchors
        """
        # None添加维度 所有target(gt)的wh wh[:, None] [6071, 2] -> [6071, 1, 2]
        #             所有anchor的wh k[None] [9, 2] -> [1, 9, 2]
        # r: target的高h宽w与anchor的高h_a宽w_a的比值，即h / h_a，w / w_a [6071, 9, 2] 有可能大于1，也可能小于等于1
        r = wh[:, None] / k[None]
        # x 高宽与anchor高宽比值的较小值（匹配较差） 无论r大于1还是小于1最后统一结果都要小于1 [6071, 9]
        # torch.min(r, 1 / r)保证小于1，torch.min(r, 1 / r).min(2)找到更接近于1的比值
        x = torch.min(r, 1 / r).min(2)[0]  # ratio metric
        # [6071] 为每个gt框选择匹配所有anchors宽高比例值最好的那一比值，考虑可以将宽高比值平均比照
        best = x.max(1)[0]  # best_xl
        # 每个target平均有多少个anchors满足条件
        # thr为4时bpr: 1.64009，为2时bpr: 0.86641；k-means++后为
        aat = (x > 1 / thr).float().sum(1).mean()  # anchors above threshold 当axis=1时，求的是每一行元素的和
        # 最多能被召回(通过thr)的gt框数量 / 所有gt框数量 [1]
        # 举例，thr为4的时候，得到结果是比例，意味着所有gt有多少是包含在所有anchor的1/4 - 4倍范围内的，thr为4时bpr: 1.0，为2时bpr: 0.84665
        bpr = (best > 1 / thr).float().mean()  # best possible recall
        return bpr, aat

    stride = m.stride.to(m.anchors.device).view(-1, 1, 1)  # model strides
    # 所有anchors的宽高 基于缩放后的图片大小(较长边为640 较小边相应缩放)
    anchors = m.anchors.clone() * stride  # current anchors
    # 计算出数据集所有图片的wh和当前所有anchors的bpr和aat
    # bpr: bpr(best possible recall): 最多能被召回(通过thr)的gt框数量 / 所有gt框数量  [1] 0.96223  小于0.98 才会用k-means计算anchor
    # aat(anchors above thr): [1] 3.54360 通过阈值的anchor个数
    bpr, aat = metric(anchors.cpu().view(-1, 2))
    s = f'\n{PREFIX}{aat:.2f} anchors/target, {bpr:.3f} Best Possible Recall (BPR). '
    if bpr > 0.98:  # threshold to recompute
        LOGGER.info(f'{s}Current anchors are a good fit to dataset ✅')
    else:
        LOGGER.info(f'{s}Anchors are a poor fit to dataset ⚠️, attempting to improve...')
        na = m.anchors.numel() // 2  # number of anchors
        # 如果bpr不满足要求，使用k-means + 遗传进化算法选择出与数据集更匹配的anchors框 [9, 2]
        anchors = kmean_anchors(dataset, n=na, img_size=imgsz, thr=thr, gen=1000, verbose=False)
        # 计算新的anchors的bpr
        new_bpr = metric(anchors)[0]
        if new_bpr > bpr:  # replace anchors
            anchors = torch.tensor(anchors, device=m.anchors.device).type_as(m.anchors)
            m.anchors[:] = anchors.clone().view_as(m.anchors)
            check_anchor_order(m)  # must be in pixel-space (not grid-space)
            m.anchors /= stride
            s = f'{PREFIX}Done ✅ (optional: update model *.yaml to use these anchors in the future)'
        else:
            s = f'{PREFIX}Done ⚠️ (original anchors better than new anchors, proceeding with original anchors)'
        LOGGER.info(s)


def kmean_anchors(dataset='./data/coco128.yaml', n=9, img_size=640, thr=4.0, gen=1000, verbose=True):
    """ Creates kmeans-evolved anchors from training dataset

        Arguments:
            dataset: path to data.yaml, or a loaded dataset
            n: number of anchors
            img_size: image size used for training
            thr: anchor-label wh ratio threshold hyperparameter hyp['anchor_t'] used for training, default=4.0
            gen: generations to evolve anchors using genetic algorithm
            verbose: print all results

        Return:
            k: kmeans evolved anchors

        Usage:
            from utils.autoanchor import *; _ = kmean_anchors()
    """
    from scipy.cluster.vq import kmeans

    npr = np.random
    thr = 1 / thr

    def metric(k, wh):  # compute metrics
        r = wh[:, None] / k[None]
        x = torch.min(r, 1 / r).min(2)[0]  # ratio metric
        # x = wh_iou(wh, torch.tensor(k))  # iou metric
        return x, x.max(1)[0]  # x, best_x

    def anchor_fitness(k):  # mutation fitness
        _, best = metric(torch.tensor(k, dtype=torch.float32), wh)
        return (best * (best > thr).float()).mean()  # fitness

    def print_results(k, verbose=True):
        k = k[np.argsort(k.prod(1))]  # sort small to large
        x, best = metric(k, wh0)
        bpr, aat = (best > thr).float().mean(), (x > thr).float().mean() * n  # best possible recall, anch > thr
        s = f'{PREFIX}thr={thr:.2f}: {bpr:.4f} best possible recall, {aat:.2f} anchors past thr\n' \
            f'{PREFIX}n={n}, img_size={img_size}, metric_all={x.mean():.3f}/{best.mean():.3f}-mean/best, ' \
            f'past_thr={x[x > thr].mean():.3f}-mean: '
        for x in k:
            s += '%i,%i, ' % (round(x[0]), round(x[1]))
        if verbose:
            LOGGER.info(s[:-2])
        return k

    if isinstance(dataset, str):  # *.yaml file
        with open(dataset, errors='ignore') as f:
            data_dict = yaml.safe_load(f)  # model dict
        from utils.dataloaders import LoadImagesAndLabels
        dataset = LoadImagesAndLabels(data_dict['train'], augment=True, rect=True)

    # Get label wh
    shapes = img_size * dataset.shapes / dataset.shapes.max(1, keepdims=True)
    wh0 = np.concatenate([l[:, 3:5] * s for s, l in zip(shapes, dataset.labels)])  # wh

    # Filter
    i = (wh0 < 3.0).any(1).sum()
    if i:
        LOGGER.info(f'{PREFIX}WARNING ⚠️ Extremely small objects found: {i} of {len(wh0)} labels are <3 pixels in size')
    wh = wh0[(wh0 >= 2.0).any(1)].astype(np.float32)  # filter > 2 pixels
    # wh = wh * (npr.rand(wh.shape[0], 1) * 0.9 + 0.1)  # multiply by random scale 0-1

    # Kmeans init
    try:
        LOGGER.info(f'{PREFIX}Running kmeans for {n} anchors on {len(wh)} points...')
        assert n <= len(wh)  # apply overdetermined constraint
        s = wh.std(0)  # sigmas for whitening
        k = kmeans(wh / s, n, iter=30)[0] * s  # points
        assert n == len(k)  # kmeans may return fewer points than requested if wh is insufficient or too similar
    except Exception:
        LOGGER.warning(f'{PREFIX}WARNING ⚠️ switching strategies from kmeans to random init')
        k = np.sort(npr.rand(n * 2)).reshape(n, 2) * img_size  # random init
    wh, wh0 = (torch.tensor(x, dtype=torch.float32) for x in (wh, wh0))
    k = print_results(k, verbose=False)

    # Plot
    # k, d = [None] * 20, [None] * 20
    # for i in tqdm(range(1, 21)):
    #     k[i-1], d[i-1] = kmeans(wh / s, i)  # points, mean distance
    # fig, ax = plt.subplots(1, 2, figsize=(14, 7), tight_layout=True)
    # ax = ax.ravel()
    # ax[0].plot(np.arange(1, 21), np.array(d) ** 2, marker='.')
    # fig, ax = plt.subplots(1, 2, figsize=(14, 7))  # plot wh
    # ax[0].hist(wh[wh[:, 0]<100, 0],400)
    # ax[1].hist(wh[wh[:, 1]<100, 1],400)
    # fig.savefig('wh.png', dpi=200)

    # Evolve
    f, sh, mp, s = anchor_fitness(k), k.shape, 0.9, 0.1  # fitness, generations, mutation prob, sigma
    pbar = tqdm(range(gen), bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}')  # progress bar
    for _ in pbar:
        v = np.ones(sh)
        while (v == 1).all():  # mutate until a change occurs (prevent duplicates)
            v = ((npr.random(sh) < mp) * random.random() * npr.randn(*sh) * s + 1).clip(0.3, 3.0)
        kg = (k.copy() * v).clip(min=2.0)
        fg = anchor_fitness(kg)
        if fg > f:
            f, k = fg, kg.copy()
            pbar.desc = f'{PREFIX}Evolving anchors with Genetic Algorithm: fitness = {f:.4f}'
            if verbose:
                print_results(k, verbose)

    return print_results(k).astype(np.float32)
