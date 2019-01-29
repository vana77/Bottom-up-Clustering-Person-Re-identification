from __future__ import print_function, absolute_import
import os.path as osp
import os
from ..utils.data import Dataset
from ..utils.osutils import mkdir_if_missing
from ..utils.serialization import write_json


class Mars(Dataset):
    def __init__(self, root, split_id=0, num_val=100, download=True):
        super(self.__class__, self).__init__(root, split_id=split_id)
        self.name="mars"
        self.num_cams = 6
        self.is_video = True

        if download:
            self.download()

        if not self._check_integrity():
            raise RuntimeError("Dataset not found or corrupted. " +
                               "You can use download=True to download it.")

        self.load(num_val)

    def download(self):
        if self._check_integrity():
            print("Files already downloaded and verified")
            return
        print("create new dataset")
        import re
        import hashlib
        import shutil
        from glob import glob
        from zipfile import ZipFile

        
        # get mars dataset

        # Format
        images_dir = osp.join(self.root, 'images')
        mkdir_if_missing(images_dir)

        # totally 1261 person (625+636) with 6 camera views each
        # id 1~625 are for training
        # id 999~1634 are for testing
        identities = [[{} for _ in range(6)] for _ in range(1635)]

        def register(subdir):
            pids = set()
            vids = []
            person_list = os.listdir(os.path.join(self.root, subdir)); person_list.sort()
            for person_id in person_list:
                count = 0
                videos = os.listdir(os.path.join(self.root, subdir, person_id)); videos.sort()
                for video_id in videos:
                    video_path = os.path.join(self.root, subdir, person_id, video_id)
                    video_id = int(video_id) - 1
                    fnames = os.listdir(video_path)
                    frame_list = []
                    for fname in fnames:
                        count += 1
                        pid = int(person_id)
                        cam = int(fname[5]) - 1
                        assert 0 <= pid <= 1634  # pid == 999, 1000 means background and distractors
                        assert 0 <= cam <= 5
                        pids.add(pid)
                        newname = ('{:04d}_{:02d}_{:04d}_{:04d}.jpg'.format(pid, cam, video_id, len(frame_list)))
                        frame_list.append(newname)
                        shutil.copy(osp.join(video_path, fname), osp.join(images_dir, newname))
                    identities[pid][cam][video_id] = frame_list
                    vids.append(frame_list)
                print("ID {}, frames {}\t  in {}".format(person_id, count, subdir))
            return pids, vids

        print("begin to preprocess mars dataset")
        trainval_pids, _ = register('train_split')
        gallery_pids, gallery_vids = register('gallery_split')
        query_pids, query_vids = register('query_split')
        #assert query_pids <= gallery_pids
        assert trainval_pids.isdisjoint(gallery_pids)

        # Save meta information into a json file
        meta = {'name': 'Mars', 'shot': 'multiple', 'num_cameras': 6,
                'identities': identities,
                'query': query_vids,
                'gallery': gallery_vids}
        write_json(meta, osp.join(self.root, 'meta.json'))

        # Save the only training / test split
        splits = [{
            'train': sorted(list(trainval_pids)),
            'query': sorted(list(query_pids)) ,
            'gallery': sorted(list(gallery_pids))}]
        write_json(splits, osp.join(self.root, 'splits.json'))

