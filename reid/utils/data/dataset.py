from __future__ import print_function
import os.path as osp

import numpy as np

from ..serialization import read_json


def _pluck(identities, indices, relabel=False):
    ret = []
    for index, pid in enumerate(indices):
        pid_images = identities[pid]
        for camid, video_ids in enumerate(pid_images):
            for video_id in video_ids:
                images = video_ids[video_id]
                if relabel:
                    ret.append((tuple(images), index, camid, video_id))
                else:
                    ret.append((tuple(images), pid, camid, video_id))
    return ret


class Dataset(object):
    def __init__(self, root, split_id=0):
        self.root = root
        self.split_id = split_id
        self.meta = None
        self.split = None
        self.train = []
        self.query, self.gallery = [], []
        self.num_train_ids = 0

    @property
    def images_dir(self):
        return osp.join(self.root, 'images')

    def load(self, verbose=True):
        splits = read_json(osp.join(self.root, 'splits.json'))
        if self.split_id >= len(splits):
            raise ValueError("split_id exceeds total splits {}"
                             .format(len(splits)))
        self.split = splits[self.split_id]

        train_pids = np.asarray(self.split['train'])


        self.meta = read_json(osp.join(self.root, 'meta.json'))
        identities = self.meta['identities']

        self.train = _pluck(identities, train_pids, relabel=True)
        self.query = _pluck(identities, self.split['query'])
        self.gallery = _pluck(identities, self.split['gallery'])
        self.num_train_ids = len(train_pids)

        if 'query' in self.meta:               
            query_fnames = self.meta['query']
            gallery_fnames = self.meta['gallery']
            self.query = []
            for fname_list in query_fnames:
                name = osp.splitext(fname_list[0])[0]
                pid, cam, vid,  _ = map(int, name.split('_'))
                self.query.append((tuple(fname_list), pid, cam, vid))
            self.gallery = []
            for fname_list in gallery_fnames:
                name = osp.splitext(fname_list[0])[0]
                pid, cam, vid, _ = map(int, name.split('_'))
                self.gallery.append((tuple(fname_list), pid, cam, vid))



        if verbose:
            print(self.__class__.__name__, "dataset loaded")
            print("  subset   | # ids | # tracklets")
            print("  ---------------------------")
            print("  train    | {:5d} | {:8d}"
                  .format(self.num_train_ids, len(self.train)))
            print("  query    | {:5d} | {:8d}"
                  .format(len(self.split['query']), len(self.query)))
            print("  gallery  | {:5d} | {:8d}"
                  .format(len(self.split['gallery']), len(self.gallery)))
            print()

    def _check_integrity(self):
        return osp.isdir(osp.join(self.root, 'images')) and \
               osp.isfile(osp.join(self.root, 'meta.json')) and \
               osp.isfile(osp.join(self.root, 'splits.json'))
