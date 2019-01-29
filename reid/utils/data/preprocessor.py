from __future__ import absolute_import
import os.path as osp

from PIL import Image
import numpy as np
import torch
import random


class Preprocessor(object):
    def __init__(self, dataset, root=None, transform=None, num_samples=16, is_training=True, max_frames = 900):
        super(Preprocessor, self).__init__()
        self.dataset = dataset
        self.root = root
        self.transform = transform
        self.selected_frames_num = num_samples
        self.is_training = is_training
        self.max_frames=max_frames

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, indices):
        if isinstance(indices, (tuple, list)):
            return [self._get_single_item(index) for index in indices]
        return self._get_single_item(indices)

    def _get_single_item(self, index):
        images, pid, camid, videoid = self.dataset[index]
        image_str = "".join(images)

        # random select images if training
        if self.is_training:
            if len(images) >= self.selected_frames_num:
                images = random.sample(images, self.selected_frames_num)
            else:
                images = random.choices(images,  k=self.selected_frames_num)
            images.sort()       

        else: # for evaluation, we use all the frames 
            if len(images) > self.max_frames:  # to avoid the insufficient memory
                images = random.sample(images, self.max_frames)

        video_frames = []
        for fname in images:
            if self.root is not None:
                fpath = osp.join(self.root, fname)
            img = Image.open(fpath).convert('RGB')
            if self.transform is not None:
                img = self.transform(img)
            video_frames.append(img)

        video_frames = torch.stack(video_frames, dim=0)
        pid = int(pid)
        return video_frames, image_str, pid, index, videoid
