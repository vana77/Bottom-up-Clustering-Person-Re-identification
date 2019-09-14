import torch
from torch import nn
from reid import models
from reid.trainers import Trainer
from reid.evaluators import extract_features, Evaluator
from reid.dist_metric import DistanceMetric
import numpy as np
from collections import OrderedDict
import os.path as osp
import pickle
import copy, sys
from reid.utils.serialization import load_checkpoint
from reid.utils.data import transforms as T
from torch.utils.data import DataLoader
from reid.utils.data.preprocessor import Preprocessor
import random
import pickle as pkl
from reid.exclusive_loss import ExLoss


class Bottom_up():
    def __init__(self, model_name, batch_size, num_classes, dataset, u_data, save_path, embeding_fea_size=1024,
                 dropout=0.5, max_frames=900, initial_steps=20, step_size=16):

        self.model_name = model_name
        self.num_classes = num_classes
        self.data_dir = dataset.images_dir
        self.is_video = dataset.is_video
        self.save_path = save_path

        self.dataset = dataset
        self.u_data = u_data
        self.u_label = np.array([label for _, label, _, _ in u_data])

        self.dataloader_params = {}
        self.dataloader_params['height'] = 256
        self.dataloader_params['width'] = 128
        self.dataloader_params['batch_size'] = batch_size
        self.dataloader_params['workers'] = 6

        self.batch_size = batch_size
        self.data_height = 256
        self.data_width = 128
        self.data_workers = 6

        self.initial_steps = initial_steps
        self.step_size = step_size

        # batch size for eval mode. Default is 1.
        self.dropout = dropout
        self.max_frames = max_frames
        self.embeding_fea_size = embeding_fea_size

        if self.is_video:
            self.eval_bs = 1
            self.fixed_layer = True
            self.frames_per_video = 16
            self.later_steps = 5
        else:
            self.eval_bs = 64
            self.fixed_layer = False
            self.frames_per_video = 1
            self.later_steps = 2

        model = models.create(self.model_name, dropout=self.dropout, 
                              embeding_fea_size=self.embeding_fea_size, fixed_layer=self.fixed_layer)
        self.model = nn.DataParallel(model).cuda()

        self.criterion = ExLoss(self.embeding_fea_size, self.num_classes, t=10).cuda()


    def get_dataloader(self, dataset, training=False):
        normalizer = T.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        if training:
            transformer = T.Compose([
                T.RandomSizedRectCrop(self.data_height, self.data_width),
                T.RandomHorizontalFlip(),
                T.ToTensor(),
                normalizer,
            ])
            batch_size = self.batch_size
        else:
            transformer = T.Compose([
                T.RectScale(self.data_height, self.data_width),
                T.ToTensor(),
                normalizer,
            ])
            batch_size = self.eval_bs
        data_dir = self.data_dir

        data_loader = DataLoader(
            Preprocessor(dataset, root=data_dir, num_samples=self.frames_per_video,
                         transform=transformer, is_training=training, max_frames=self.max_frames),
            batch_size=batch_size, num_workers=self.data_workers,
            shuffle=training, pin_memory=True, drop_last=training)

        current_status = "Training" if training else "Testing"
        print("Create dataloader for {} with batch_size {}".format(current_status, batch_size))
        return data_loader

    def train(self, train_data, step, loss, dropout=0.5):
        # adjust training epochs and learning rate
        epochs = self.initial_steps if step==0 else self.later_steps
        init_lr = 0.1 if step==0 else 0.01 
        step_size = self.step_size if step==0 else sys.maxsize

        """ create model and dataloader """
        dataloader = self.get_dataloader(train_data, training=True)

        # the base parameters for the backbone (e.g. ResNet50)
        base_param_ids = set(map(id, self.model.module.CNN.base.parameters()))

        # we fixed the first three blocks to save GPU memory
        base_params_need_for_grad = filter(lambda p: p.requires_grad, self.model.module.CNN.base.parameters())

        # params of the new layers
        new_params = [p for p in self.model.parameters() if id(p) not in base_param_ids]

        # set the learning rate for backbone to be 0.1 times
        param_groups = [
            {'params': base_params_need_for_grad, 'lr_mult': 0.1},
            {'params': new_params, 'lr_mult': 1.0}]

        optimizer = torch.optim.SGD(param_groups, lr=init_lr, momentum=0.9, weight_decay=5e-4, nesterov=True)

        # change the learning rate by step
        def adjust_lr(epoch, step_size):
            lr = init_lr / (10 ** (epoch // step_size))
            for g in optimizer.param_groups:
                g['lr'] = lr * g.get('lr_mult', 1)

        """ main training process """
        trainer = Trainer(self.model, self.criterion, fixed_layer=self.fixed_layer)
        for epoch in range(epochs):
            adjust_lr(epoch, step_size)
            trainer.train(epoch, dataloader, optimizer, print_freq=max(5, len(dataloader) // 30 * 10))

    def get_feature(self, dataset):
        dataloader = self.get_dataloader(dataset, training=False)
        features, _, fcs = extract_features(self.model, dataloader)
        features = np.array([logit.numpy() for logit in features.values()])
        fcs = np.array([logit.numpy() for logit in fcs.values()])
        return features, fcs

    def update_memory(self, weight):
        self.criterion.weight = torch.from_numpy(weight).cuda()

    def evaluate(self, query, gallery):
        test_loader = self.get_dataloader(list(set(query) | set(gallery)), training=False)
        evaluator = Evaluator(self.model)
        rank1, mAP = evaluator.evaluate(test_loader, query, gallery)
        return rank1, mAP

    def calculate_distance(self,u_feas):
        # calculate distance between features
        x = torch.from_numpy(u_feas)
        y = x
        m = len(u_feas)
        dists = torch.pow(x, 2).sum(dim=1, keepdim=True).expand(m, m) + \
                torch.pow(y, 2).sum(dim=1, keepdim=True).expand(m, m).t()
        dists.addmm_(1, -2, x, y.t())
        return dists

    # def select_merge_data(self, u_feas, nums_to_merge, label, label_to_images,  ratio_n,  dists):
    #     #calculate final distance (feature distance + diversity regularization)
    #     tri = np.tri(len(u_feas), dtype=np.float32)
    #     tri = tri * np.iinfo(np.int32).max
    #     tri = tri.astype('float32')
    #     tri = torch.from_numpy(tri)
    #     dists = dists + tri
    #     for idx in range(len(u_feas)):
    #         for j in range(idx + 1, len(u_feas)):
    #             if label[idx] == label[j]:
    #                 dists[idx, j] = np.iinfo(np.int32).max
    #             else:
    #                 dists[idx][j] =  dists[idx][j] + \
    #                                 + ratio_n * ((len(label_to_images[label[idx]])) + (len(label_to_images[label[j]])))
    #     dists = dists.numpy()
    #     ind = np.unravel_index(np.argsort(dists, axis=None), dists.shape)
    #     idx1 = ind[0]
    #     idx2 = ind[1]
    #     return idx1, idx2


    def select_merge_data(self, u_feas, label, label_to_images,  ratio_n,  dists):
        dists.add_(torch.tril(100000 * torch.ones(len(u_feas), len(u_feas))))

        cnt = torch.FloatTensor([ len(label_to_images[label[idx]]) for idx in range(len(u_feas))])
        dists += ratio_n * (cnt.view(1, len(cnt)) + cnt.view(len(cnt), 1))
        
        for idx in range(len(u_feas)):
            for j in range(idx + 1, len(u_feas)):
                if label[idx] == label[j]:
                    dists[idx, j] = 100000

        dists = dists.numpy()
        ind = np.unravel_index(np.argsort(dists, axis=None), dists.shape)
        idx1 = ind[0]
        idx2 = ind[1]
        return idx1, idx2



    def generate_new_train_data(self, idx1, idx2, label,num_to_merge):
        correct = 0
        num_before_merge = len(np.unique(np.array(label)))
        # merge clusters with minimum dissimilarity
        for i in range(len(idx1)):
            label1 = label[idx1[i]]
            label2 = label[idx2[i]]
            if label1 < label2:
                label = [label1 if x == label2 else x for x in label]
            else:
                label = [label2 if x == label1 else x for x in label]
            if self.u_label[idx1[i]] == self.u_label[idx2[i]]:
                correct += 1
            num_merged =  num_before_merge - len(np.sort(np.unique(np.array(label))))
            if num_merged == num_to_merge:
                break

        # set new label to the new training data
        unique_label = np.sort(np.unique(np.array(label)))
        for i in range(len(unique_label)):
            label_now = unique_label[i]
            label = [i if x == label_now else x for x in label]
        new_train_data = []
        for idx, data in enumerate(self.u_data):
            new_data = copy.deepcopy(data)
            new_data[3] = label[idx]
            new_train_data.append(new_data)

        num_after_merge = len(np.unique(np.array(label)))
        print("num of label before merge: ", num_before_merge, " after_merge: ", num_after_merge, " sub: ",
              num_before_merge - num_after_merge)
        return new_train_data, label

    def generate_average_feature(self, labels):
        #extract feature/classifier
        u_feas, fcs = self.get_feature(self.u_data)

        #images of the same cluster
        label_to_images = {}
        for idx, l in enumerate(labels):
            label_to_images[l] = label_to_images.get(l, []) + [idx]

        #calculate average feature/classifier of a cluster
        feature_avg = np.zeros((len(label_to_images), len(u_feas[0])))
        fc_avg = np.zeros((len(label_to_images), len(fcs[0])))
        for l in label_to_images:
            feas = u_feas[label_to_images[l]]
            feature_avg[l] = np.mean(feas, axis=0)
            fc_avg[l] = np.mean(fcs[label_to_images[l]], axis=0)
        return u_feas, feature_avg, label_to_images, fc_avg

    def get_new_train_data(self, labels, nums_to_merge, size_penalty):
        u_feas, feature_avg, label_to_images, fc_avg = self.generate_average_feature(labels)
        
        dists = self.calculate_distance(u_feas)
        
        idx1, idx2 = self.select_merge_data(u_feas, labels, label_to_images, size_penalty,dists)
        
        new_train_data, labels = self.generate_new_train_data(idx1, idx2, labels,nums_to_merge)
        
        num_train_ids = len(np.unique(np.array(labels)))

        # change the criterion classifer
        self.criterion = ExLoss(self.embeding_fea_size, num_train_ids, t=10).cuda()
        #new_classifier = fc_avg.astype(np.float32)
        #self.criterion.V = torch.from_numpy(new_classifier).cuda()

        return labels, new_train_data


def change_to_unlabel(dataset):
    # generate unlabeled set
    trimmed_dataset = []
    init_videoid = int(dataset.train[0][3])
    for (imgs, pid, camid, videoid) in dataset.train:
        videoid = int(videoid) - init_videoid
        if videoid < 0:
            print(videoid, 'RANGE ERROR')
        assert videoid >= 0
        trimmed_dataset.append([imgs, pid, camid, videoid])

    index_labels = []
    for idx, data in enumerate(trimmed_dataset):
        data[3] = idx # data[3] is the label of the data array
        index_labels.append(data[3])  # index
    
    return trimmed_dataset, index_labels
