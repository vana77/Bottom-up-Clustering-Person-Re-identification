from __future__ import print_function, absolute_import
import time
from collections import OrderedDict

import torch

from .evaluation_metrics import cmc, mean_ap
from .feature_extraction import extract_cnn_feature
from .utils.meters import AverageMeter


from torch.backends import cudnn

def extract_features(model, data_loader, print_freq=1, metric=None):
    cudnn.benchmark = False
    model.eval()
    batch_time = AverageMeter()
    data_time = AverageMeter()

    features = OrderedDict()
    labels = OrderedDict()
    fcs = OrderedDict()

    print("Begin to extract features...")
    for i, (imgs, fnames, pids, _, _) in enumerate(data_loader):
        _fcs, pool5s = extract_cnn_feature(model, imgs)
        for fname, fc, pool5, pid in zip(fnames, _fcs, pool5s, pids):
            features[fname] = pool5
            fcs[fname] = fc
            labels[fname] = pid
            
    cudnn.benchmark = True
    return features, labels, fcs


def pairwise_distance(features, query=None, gallery=None, metric=None):
    if query is None and gallery is None:
        n = len(features)
        x = torch.cat(list(features.values()))
        x = x.view(n, -1)
        if metric is not None:
            x = metric.transform(x)
        dist = torch.pow(x, 2).sum(dim=1, keepdim=True) * 2
        dist = dist.expand(n, n) - 2 * torch.mm(x, x.t())
        return dist

    x = torch.cat([features["".join(f)].unsqueeze(0) for f, _, _, _ in query], 0)
    y = torch.cat([features["".join(f)].unsqueeze(0) for f, _, _, _ in gallery], 0)
    m, n = x.size(0), y.size(0)
    x = x.view(m, -1)
    y = y.view(n, -1)
    if metric is not None:
        x = metric.transform(x)
        y = metric.transform(y)
    dist = torch.pow(x, 2).sum(dim=1, keepdim=True).expand(m, n) + \
           torch.pow(y, 2).sum(dim=1, keepdim=True).expand(n, m).t()
    dist.addmm_(1, -2, x, y.t())
    return dist


def evaluate_all(distmat, query=None, gallery=None,
                 query_ids=None, gallery_ids=None,
                 query_cams=None, gallery_cams=None,
                 cmc_topk=(1, 5, 10, 20)):
    if query is not None and gallery is not None:
        query_ids = [pid for _, pid, _, _ in query]
        gallery_ids = [pid for _, pid, _, _ in gallery]
        query_cams = [cam for _, _, cam, _ in query]
        gallery_cams = [cam for _, _, cam, _ in gallery]
    else:
        assert (query_ids is not None and gallery_ids is not None
                and query_cams is not None and gallery_cams is not None)

    # Compute mean AP
    mAP = mean_ap(distmat, query_ids, gallery_ids, query_cams, gallery_cams)        

    # Compute all kinds of CMC scores
    cmc_configs = {
        'market1501': dict(separate_camera_set=False,
                           single_gallery_shot=False,
                           first_match_break=True)}
    cmc_scores = {name: cmc(distmat, query_ids, gallery_ids,
                            query_cams, gallery_cams, **params)
                  for name, params in cmc_configs.items()}

    print('Mean AP: {:4.1%}'.format(mAP))
    print('CMC Scores:')
    for k in cmc_topk:
        print('  top-{:<4}{:12.1%}'
              .format(k, 
                      cmc_scores['market1501'][k - 1]))

    # Use the allshots cmc top-1 score for validation criterion
    return cmc_scores['market1501'][0], mAP


class Evaluator(object):
    def __init__(self, model, is_print=False):
        super(Evaluator, self).__init__()
        self.model = model
    def evaluate(self, data_loader, query, gallery, metric=None):
        features, _, _ = extract_features(self.model, data_loader)
        distmat = pairwise_distance(features, query, gallery, metric=metric)
        return evaluate_all(distmat, query=query, gallery=gallery)
