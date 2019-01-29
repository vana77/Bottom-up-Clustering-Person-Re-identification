from __future__ import print_function, absolute_import
import time

import torch
from torch.autograd import Variable

from .evaluation_metrics import accuracy
from .utils.meters import AverageMeter


class BaseTrainer(object):
    def __init__(self, model, criterion, fixed_layer=True):
        super(BaseTrainer, self).__init__()
        self.model = model
        self.criterion = criterion
        self.fixed_layer = fixed_layer

    def train(self, epoch, data_loader, optimizer, print_freq=1):
        self.model.train()

        if self.fixed_layer:
            # The following code is used to keep the BN on the first three block fixed 
            fixed_bns = []
            for idx, (name, module) in enumerate(self.model.module.named_modules()):
                if name.find("layer3") != -1:
                    assert len(fixed_bns) == 22
                    break
                if name.find("bn") != -1:
                    fixed_bns.append(name)
                    module.eval() 

        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter()
        precisions = AverageMeter()

        end = time.time()
        for i, inputs in enumerate(data_loader):
            data_time.update(time.time() - end)

            inputs, targets = self._parse_data(inputs)
            loss, prec1 = self._forward(inputs, targets)
    
            losses.update(loss.item(), targets.size(0))
            precisions.update(prec1, targets.size(0))

            optimizer.zero_grad()
            loss.backward()

            # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.
            #torch.nn.utils.clip_grad_norm(self.model.parameters(), 0.75)
            optimizer.step()

            batch_time.update(time.time() - end)
            end = time.time()

            if (i + 1) % print_freq == 0:
                print('Epoch: [{}][{}/{}]\t'
                      'Time {:.3f} ({:.3f})\t'
                      'Data {:.3f} ({:.3f})\t'
                      'Loss {:.3f} ({:.3f})\t'
                      'Prec {:.2%} ({:.2%})\t'
                      .format(epoch, i + 1, len(data_loader),
                              batch_time.val, batch_time.avg,
                              data_time.val, data_time.avg,
                              losses.val, losses.avg,
                              precisions.val, precisions.avg))

    def _parse_data(self, inputs):
        raise NotImplementedError

    def _forward(self, inputs, targets):
        raise NotImplementedError


class Trainer(BaseTrainer):
    def _parse_data(self, inputs):
        imgs, _, pids, indexs, videoid = inputs
        inputs = Variable(imgs, requires_grad=False)
        targets = Variable(videoid.cuda())
        return inputs, targets

    def _forward(self, inputs, targets):
        outputs, _ = self.model(inputs)
        loss, outputs = self.criterion(outputs, targets)
        prec, = accuracy(outputs.data, targets.data)
        prec = prec[0]
        return loss, prec

