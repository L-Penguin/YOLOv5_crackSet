# YOLOv5 🚀 by Ultralytics, GPL-3.0 license
"""
Dataloaders
"""

import os
import random

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader, distributed

from ..augmentations import augment_hsv, copy_paste, letterbox
from ..dataloaders import InfiniteDataLoader, LoadImagesAndLabels, seed_worker
from ..general import LOGGER, xyn2xy, xywhn2xyxy, xyxy2xywhn
from ..torch_utils import torch_distributed_zero_first
from .augmentations import mixup, random_perspective

import torch.nn.functional as F

RANK = int(os.getenv('RANK', -1))


def create_dataloader(path,
                      imgsz,
                      batch_size,
                      stride,
                      single_cls=False,
                      hyp=None,
                      augment=False,
                      cache=False,
                      pad=0.0,
                      rect=False,
                      rank=-1,
                      workers=8,
                      image_weights=False,
                      quad=False,
                      prefix='',
                      shuffle=False,
                      mask_downsample_ratio=1,
                      overlap_mask=False,
                      saveMosaicImg=False,
                      concatSet=False):
    if rect and shuffle:
        LOGGER.warning('WARNING ⚠️ --rect is incompatible with DataLoader shuffle, setting shuffle=False')
        shuffle = False
    with torch_distributed_zero_first(rank):  # init dataset *.cache only once if DDP
        dataset = LoadImagesAndLabelsAndMasks(
            path,
            imgsz,
            batch_size,
            augment=augment,  # augmentation
            hyp=hyp,  # hyperparameters
            rect=rect,  # rectangular batches
            cache_images=cache,
            single_cls=single_cls,
            stride=int(stride),
            pad=pad,
            image_weights=image_weights,
            prefix=prefix,
            downsample_ratio=mask_downsample_ratio,
            overlap=overlap_mask,
            saveMosaicImg=saveMosaicImg,
            concatSet=concatSet)

    batch_size = min(batch_size, len(dataset))
    nd = torch.cuda.device_count()  # number of CUDA devices
    nw = min([os.cpu_count() // max(nd, 1), batch_size if batch_size > 1 else 0, workers])  # number of workers
    sampler = None if rank == -1 else distributed.DistributedSampler(dataset, shuffle=shuffle)
    loader = DataLoader if image_weights else InfiniteDataLoader  # only DataLoader allows for attribute updates
    generator = torch.Generator()
    generator.manual_seed(6148914691236517205 + RANK)
    return loader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle and sampler is None,
        num_workers=0,
        sampler=sampler,
        pin_memory=True,
        collate_fn=LoadImagesAndLabelsAndMasks.collate_fn4 if quad else LoadImagesAndLabelsAndMasks.collate_fn,
        worker_init_fn=seed_worker,
        generator=generator,
    ), dataset


class LoadImagesAndLabelsAndMasks(LoadImagesAndLabels):  # for training/testing

    def __init__(
        self,
        path,
        img_size=640,
        batch_size=16,
        augment=False,
        hyp=None,
        rect=False,
        image_weights=False,
        cache_images=False,
        single_cls=False,
        stride=32,
        pad=0,
        min_items=0,
        prefix="",
        downsample_ratio=1,
        overlap=False,
        saveMosaicImg=False,
        concatSet=False
    ):
        super().__init__(path, img_size, batch_size, augment, hyp, rect, image_weights, cache_images, single_cls,
                         stride, pad, min_items, prefix)
        self.downsample_ratio = downsample_ratio
        self.overlap = overlap
        self.saveMosaicImg = saveMosaicImg
        self.concatSet = concatSet

    def __getitem__(self, index):
        index = self.indices[index]  # linear, shuffled, or image_weights

        hyp = self.hyp
        mosaic = self.mosaic and random.random() < hyp['mosaic']
        masks = []
        if mosaic:
            # Load mosaic
            img, labels, segments = self.load_mosaic(index)
            shapes = None

            # MixUp augmentation
            if random.random() < hyp["mixup"]:
                img, labels, segments = mixup(img, labels, segments, *self.load_mosaic(random.randint(0, self.n - 1)))

        else:
            # Load image
            img, (h0, w0), (h, w) = self.load_image(index)

            # Letterbox
            shape = self.batch_shapes[self.batch[index]] if self.rect else self.img_size  # final letterboxed shape
            img, ratio, pad = letterbox(img, shape, auto=False, scaleup=self.augment)
            shapes = (h0, w0), ((h / h0, w / w0), pad)  # for COCO mAP rescaling

            labels = self.labels[index].copy()  # box标签（比例）
            # [array, array, ....], array.shape=(num_points, 2), xyxyxyxy
            segments = self.segments[index].copy()  # segment标签比例
            if len(segments):
                for i_s in range(len(segments)):
                    segments[i_s] = xyn2xy(     # segment标签扩展到指定大小图像像素位置
                        segments[i_s],
                        ratio[0] * w,
                        ratio[1] * h,
                        padw=pad[0],
                        padh=pad[1],
                    )
            if labels.size:  # normalized xywh to pixel xyxy format
                labels[:, 1:] = xywhn2xyxy(labels[:, 1:], ratio[0] * w, ratio[1] * h, padw=pad[0], padh=pad[1])

            if self.augment:
                img, labels, segments = random_perspective(img,
                                                           labels,
                                                           segments=segments,
                                                           degrees=hyp["degrees"],
                                                           translate=hyp["translate"],
                                                           scale=hyp["scale"],
                                                           shear=hyp["shear"],
                                                           perspective=hyp["perspective"])

        nl = len(labels)  # number of labels
        if nl:
            labels[:, 1:5] = xyxy2xywhn(labels[:, 1:5], w=img.shape[1], h=img.shape[0], clip=True, eps=1e-3)
            if self.overlap:
                masks, sorted_idx = polygons2masks_overlap(img.shape[:2],   # 相较于图像大小下采样
                                                           segments,
                                                           downsample_ratio=self.downsample_ratio)
                masks = masks[None]  # (640, 640) -> (1, 640, 640)
                labels = labels[sorted_idx]
            else:
                masks = polygons2masks(img.shape[:2], segments, color=1, downsample_ratio=self.downsample_ratio)

        masks = (torch.from_numpy(masks) if len(masks) else torch.zeros(1 if self.overlap else nl, img.shape[0] //
                                                                        self.downsample_ratio, img.shape[1] //
                                                                        self.downsample_ratio))
        # TODO: albumentations support
        if self.augment:
            # Albumentations
            # there are some augmentation that won't change boxes and masks,
            # so just be it for now.
            img, labels = self.albumentations(img, labels)
            nl = len(labels)  # update after albumentations

            # HSV color-space
            augment_hsv(img, hgain=hyp["hsv_h"], sgain=hyp["hsv_s"], vgain=hyp["hsv_v"])

            # Flip up-down
            if random.random() < hyp["flipud"]:
                img = np.flipud(img)
                if nl:
                    labels[:, 2] = 1 - labels[:, 2]
                    masks = torch.flip(masks, dims=[1])

            # Flip left-right
            if random.random() < hyp["fliplr"]:
                img = np.fliplr(img)
                if nl:
                    labels[:, 1] = 1 - labels[:, 1]
                    masks = torch.flip(masks, dims=[2])

            # Cutouts  # labels = cutout(img, labels, p=0.5)
        # 需要搭配no-overlap使用
        if self.concatSet:
            new_masks, new_labels = self.solve_masks_labels(masks, labels, img.shape)
            nl = len(new_labels)
        else:
            new_masks, new_labels = masks, labels

        if self.saveMosaicImg:
            # 保存增强之后的图像
            self.save_mosaic_images(img, new_masks, new_labels, index)

        labels_out = torch.zeros((nl, 6))
        if nl:
            labels_out[:, 1:] = torch.from_numpy(new_labels)

        # Convert
        img = img.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
        img = np.ascontiguousarray(img)

        return torch.from_numpy(img), labels_out, self.im_files[index], shapes, new_masks

    # 解决标签问题
    def solve_masks_labels(self, masks, labels, shape):
        # 将遮盖层和标签进行处理，相接的遮盖层连接在一起并将标签合并
        masks_float32 = masks.type(torch.float32)
        try:
            masks_numpy = F.interpolate(masks_float32[None],
                                        shape[:2],
                                        mode='bilinear',
                                        align_corners=False)[0].cpu().numpy()
        except RuntimeError:
            masks_numpy = masks_float32.reshape((-1, shape[0], shape[1]))

        kernel = np.ones((5, 5))
        for i, m in enumerate(masks_numpy):
            masks_numpy[i] = cv2.filter2D(m, -1, kernel)

        connectArr = []
        masks_numpy[masks_numpy > 0] = 1
        for i in range(len(masks_numpy) - 1):
            for j in range(i + 1, len(masks_numpy)):
                if (masks_numpy[i] + masks_numpy[j]).max() == 2:
                    if len(connectArr) != 0:
                        for index, arr in enumerate(connectArr):
                            if len(set(arr + [i, j])) < len(arr) + 2:
                                connectArr[index] = list(set(arr + [i, j]))
                            else:
                                connectArr.append([i, j])
                    else:
                        connectArr.append([i, j])
        new_masks, new_labels = self.concat_masks_labels(masks.cpu().numpy(), labels.copy(), connectArr, shape)
        new_masks = torch.tensor(new_masks, dtype=torch.uint8)

        return new_masks, new_labels

    def concat_masks_labels(self, masks, labels, target, shape):
        saveIndex = np.zeros(len(masks)) == 0
        for t in target:
            # 合并mask并赋值
            masks[t[0]] = masks[t].sum(axis=0)
            masks[t[0], masks[t[0]] > 0] = 1
            # 合并labels并赋值
            labels_ = xywhn2xyxy(labels[t, 1:], shape[0], shape[1], 0, 0)
            x_labels = labels_[:, [0, 2]]
            x_min, x_max = x_labels.min(), x_labels.max()
            y_labels = labels_[:, [1, 3]]
            y_min, y_max = y_labels.min(), y_labels.max()
            labels[t[0], 1:5] = xyxy2xywhn(np.array([x_min, y_min, x_max, y_max]).reshape(1, -1),
                                           shape[0],
                                           shape[1],
                                           clip=True,
                                           eps=1e-3)

            saveIndex[t[1:]] = False
        # 去除合并所剩余的mask
        return masks[saveIndex], labels[saveIndex]

    def save_mosaic_images(self, img, masks, labels, index):
        # 存储mosaic处理的图片
        workPath = os.getcwd()
        dirName = 'mosaic_degrees'
        dirPath = os.path.join(workPath, dirName)
        if not os.path.exists(dirPath):
            os.mkdir(dirPath)
        fileName = os.path.basename(self.im_files[index])
        oriName = os.path.splitext(fileName)[0] + '_ori' + os.path.splitext(fileName)[1]
        path_1 = os.path.join(dirPath, f'{fileName}')
        path_2 = os.path.join(dirPath, f'{oriName}')
        ic_1 = img.copy()
        labels_xyxy = xywhn2xyxy(labels[:, 1:], img.shape[0], img.shape[1], 0, 0)
        # 绘制目标检测框
        for l in labels_xyxy:
            point_1 = (int(l[0]), int(l[1]))
            point_2 = (int(l[2]), int(l[3]))
            cv2.rectangle(ic_1, point_1, point_2, (255, 0, 0), 2)
        # 绘制遮盖层
        masks_ = masks.type(torch.float32)
        if not self.overlap:
            masks_ = masks.sum(axis=0, keepdim=True)
            masks_ = masks_.type(torch.float32)
        masks_ = F.interpolate(masks_[None], ic_1.shape[:2], mode='bilinear', align_corners=False)[0]
        masks_[masks_ > 1] = 1
        mask_ = masks_.permute((1, 2, 0)) * 255
        c1 = torch.zeros(mask_.shape, dtype=mask_.dtype)
        c2 = torch.zeros(mask_.shape, dtype=mask_.dtype)
        mask = torch.cat((c1, c2, mask_), dim=2).cpu().numpy()
        mask = mask.astype(np.uint8)
        output = cv2.addWeighted(ic_1, 1, mask, 0.5, 0)
        cv2.imwrite(path_1, output)
        cv2.imwrite(path_2, ic_1)

    def load_mosaic(self, index):
        # YOLOv5 4-mosaic loader. Loads 1 image + 3 random images into a 4-image mosaic
        labels4, segments4 = [], []
        s = self.img_size
        yc, xc = (int(random.uniform(-x, 2 * s + x)) for x in self.mosaic_border)  # mosaic center x, y
        indices = [index] + random.choices(self.indices, k=3)  # 3 additional image indices

        path = os.path.join(os.getcwd(), f'./mosaic_seg.txt')
        with open(path, 'a+') as f:
            content = os.path.basename(self.im_files[index]) + f':\t'
            for i in indices:
                content += os.path.basename(self.im_files[i]) + f'+'
            f.write(content + '\n')

        for i, index in enumerate(indices):
            # Load image
            img, _, (h, w) = self.load_image(index)

            # place img in img4
            if i == 0:  # top left
                img4 = np.full((s * 2, s * 2, img.shape[2]), 114, dtype=np.uint8)  # base image with 4 tiles
                x1a, y1a, x2a, y2a = max(xc - w, 0), max(yc - h, 0), xc, yc  # xmin, ymin, xmax, ymax (large image)
                x1b, y1b, x2b, y2b = w - (x2a - x1a), h - (y2a - y1a), w, h  # xmin, ymin, xmax, ymax (small image)
            elif i == 1:  # top right
                x1a, y1a, x2a, y2a = xc, max(yc - h, 0), min(xc + w, s * 2), yc
                x1b, y1b, x2b, y2b = 0, h - (y2a - y1a), min(w, x2a - x1a), h
            elif i == 2:  # bottom left
                x1a, y1a, x2a, y2a = max(xc - w, 0), yc, xc, min(s * 2, yc + h)
                x1b, y1b, x2b, y2b = w - (x2a - x1a), 0, w, min(y2a - y1a, h)
            elif i == 3:  # bottom right
                x1a, y1a, x2a, y2a = xc, yc, min(xc + w, s * 2), min(s * 2, yc + h)
                x1b, y1b, x2b, y2b = 0, 0, min(w, x2a - x1a), min(y2a - y1a, h)

            img4[y1a:y2a, x1a:x2a] = img[y1b:y2b, x1b:x2b]  # img4[ymin:ymax, xmin:xmax]
            padw = x1a - x1b
            padh = y1a - y1b

            labels, segments = self.labels[index].copy(), self.segments[index].copy()

            if labels.size:
                labels[:, 1:] = xywhn2xyxy(labels[:, 1:], w, h, padw, padh)  # normalized xywh to pixel xyxy format
                segments = [xyn2xy(x, w, h, padw, padh) for x in segments]
            labels4.append(labels)
            segments4.extend(segments)

        # Concat/clip labels
        labels4 = np.concatenate(labels4, 0)
        for x in (labels4[:, 1:], *segments4):
            np.clip(x, 0, 2 * s, out=x)  # clip when using random_perspective()
        # img4, labels4 = replicate(img4, labels4)  # replicate

        # Augment
        img4, labels4, segments4 = copy_paste(img4, labels4, segments4, p=self.hyp["copy_paste"])
        img4, labels4, segments4 = random_perspective(img4,
                                                      labels4,
                                                      segments4,
                                                      degrees=self.hyp["degrees"],
                                                      translate=self.hyp["translate"],
                                                      scale=self.hyp["scale"],
                                                      shear=self.hyp["shear"],
                                                      perspective=self.hyp["perspective"],
                                                      border=self.mosaic_border)  # border to remove
        return img4, labels4, segments4

    @staticmethod
    def collate_fn(batch):
        img, label, path, shapes, masks = zip(*batch)  # transposed
        batched_masks = torch.cat(masks, 0)
        for i, l in enumerate(label):
            l[:, 0] = i  # add target image index for build_targets()
        return torch.stack(img, 0), torch.cat(label, 0), path, shapes, batched_masks


def polygon2mask(img_size, polygons, color=1, downsample_ratio=1):
    """
    Args:
        img_size (tuple): The image size.
        polygons (np.ndarray): [N, M], N is the number of polygons,
            M is the number of points(Be divided by 2).
    """
    mask = np.zeros(img_size, dtype=np.uint8)
    polygons = np.asarray(polygons)
    polygons = polygons.astype(np.int32)
    shape = polygons.shape
    polygons = polygons.reshape(shape[0], -1, 2)
    cv2.fillPoly(mask, polygons, color=color)
    nh, nw = (img_size[0] // downsample_ratio, img_size[1] // downsample_ratio)
    # NOTE: fillPoly firstly then resize is trying the keep the same way
    # of loss calculation when mask-ratio=1.
    mask = cv2.resize(mask, (nw, nh))
    return mask


def polygons2masks(img_size, polygons, color, downsample_ratio=1):
    """
    Args:
        img_size (tuple): The image size.
        polygons (list[np.ndarray]): each polygon is [N, M],
            N is the number of polygons,
            M is the number of points(Be divided by 2).
    """
    masks = []
    for si in range(len(polygons)):
        mask = polygon2mask(img_size, [polygons[si].reshape(-1)], color, downsample_ratio)
        masks.append(mask)
    return np.array(masks)


def polygons2masks_overlap(img_size, segments, downsample_ratio=1):
    """Return a (640, 640) overlap mask."""
    masks = np.zeros((img_size[0] // downsample_ratio, img_size[1] // downsample_ratio),
                     dtype=np.int32 if len(segments) > 255 else np.uint8)
    areas = []
    ms = []
    for si in range(len(segments)):
        mask = polygon2mask(
            img_size,
            [segments[si].reshape(-1)],
            downsample_ratio=downsample_ratio,
            color=1,
        )
        ms.append(mask)
        areas.append(mask.sum())
    areas = np.asarray(areas)
    index = np.argsort(-areas)
    ms = np.array(ms)[index]
    for i in range(len(segments)):
        mask = ms[i] * (i + 1)
        masks = masks + mask
        masks = np.clip(masks, a_min=0, a_max=i + 1)
    return masks, index
