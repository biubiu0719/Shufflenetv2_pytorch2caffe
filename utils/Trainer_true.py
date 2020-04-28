from __future__ import print_function

import os
import numpy as np

import torch as t
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.autograd import Variable
from torchnet import meter

from .log import logger
from .visualize import Visualizer

#获取learning_rates
def get_learning_rates(optimizer):
    lrs = [pg['lr'] for pg in optimizer.param_groups]
    lrs = np.asarray(lrs, dtype=np.float)
    return lrs


class DataIterator(object):

    def __init__(self, dataloader):
        self.dataloader = dataloader
        self.iterator = enumerate(self.dataloader)

    def next(self):
        try:
            _, data = next(self.iterator)
        except Exception:
            self.iterator = enumerate(self.dataloader)
            _, data = next(self.iterator)
        return data[0], data[1]

#训练参数
class TrainParams(object):
    # required params
    max_epoch = 1000

    # optimizer and criterion and learning rate scheduler
    optimizer = None
    criterion = None
    lr_scheduler = None         # should be an instance of ReduceLROnPlateau or _LRScheduler

    # params based on your local env
    gpus = []  # default to use CPU mode
    save_dir = './models/shufflenetv2_huajie_clip_body_norotatee/'            # default `save_dir`

    # loading existing checkpoint
    ckpt = None                 # path to the ckpt file
    # saving checkpoints
    save_freq_epoch = 50        # save one ckpt per `save_freq_epoch` epochs
    batch_size = 1024

#主函数
class Trainer(object):

    TrainParams = TrainParams#默认参数的获取

    def __init__(self, model, model_name, train_params, train_data, val_data=None, val_data_brand_list=None, train_data_len=0, num_classes=2, visualizer_port=8899):
        assert isinstance(train_params, TrainParams)
        self.params = train_params
        self.num_classes = num_classes
        self.model_name = model_name
        self.visualizer_port = visualizer_port
        self.train_data_len = train_data_len

        # Data loaders
        self.train_data = train_data
        self.val_data = val_data
        self.val_data_brand_list = val_data_brand_list

        # criterion and Optimizer and learning rate
        self.last_epoch = 0
        self.criterion = self.params.criterion
        self.optimizer = self.params.optimizer
        self.lr_scheduler = self.params.lr_scheduler
        logger.info('Set criterion to {}'.format(type(self.criterion)))
        logger.info('Set optimizer to {}'.format(type(self.optimizer)))
        logger.info('Set lr_scheduler to {}'.format(type(self.lr_scheduler)))

        # load model
        self.model = model
        logger.info('Set output dir to {}'.format(self.params.save_dir))
        if os.path.isdir(self.params.save_dir):
            pass
        else:
            os.makedirs(self.params.save_dir)

        ckpt = self.params.ckpt
        if ckpt is not None:
            self._load_ckpt(ckpt)
            logger.info('Load ckpt from {}'.format(ckpt))

        # meters
        self.loss_meter = meter.AverageValueMeter()
        self.confusion_matrix = meter.ConfusionMeter(self.num_classes)

        # set CUDA_VISIBLE_DEVICES
        if len(self.params.gpus) > 0:
            gpus = ','.join([str(x) for x in self.params.gpus])
            os.environ['CUDA_VISIBLE_DEVICES'] = gpus
            self.params.gpus = tuple(range(len(self.params.gpus)))
            logger.info('Set CUDA_VISIBLE_DEVICES to {}...'.format(gpus))
            self.model = nn.DataParallel(self.model, device_ids=self.params.gpus)
            self.model = self.model.cuda()

        self.model.train()

    def train(self):
        vis = Visualizer(env=self.model_name, port=self.visualizer_port)
        best_loss = np.inf
        val_list = []
        val_list.append(0)
        for epoch in range(self.last_epoch, self.params.max_epoch):

            self.loss_meter.reset()
            self.confusion_matrix.reset()

            self.last_epoch += 1
            logger.info('Start training epoch {}'.format(self.last_epoch))
            #logger.info('lr is {}'.format(self.optimizer.get_lr()))

            self._train_one_epoch()
            #val_interval = int( self.train_data_len / self.params.batch_size ) * 2
            #if epoch % val_interval == 0 or epoch == self.params.max_epoch - 1 or epoch == 1:
            if True:
                val_cm, val_accuracy = self._val_one_epoch(self.val_data)

                # save model
                if (self.last_epoch % self.params.save_freq_epoch == 0) or (self.last_epoch == self.params.max_epoch - 1):
                    save_name = self.params.save_dir + 'ckpt_epoch_{}_{}.pth'.format(self.last_epoch, val_accuracy)
                    print('save model: {}'.format(save_name))
                    t.save(self.model.state_dict(), save_name)
                elif len(val_list) >= 1 and val_accuracy > val_list[-1]:
                    save_name = self.params.save_dir + 'ckpt_epoch_{}_{}.pth'.format(self.last_epoch, val_accuracy)
                    print('Found a better model save model: {}'.format(save_name))
                    t.save(self.model.state_dict(), save_name)
           

                #val_cm, val_accuracy = self._val_one_epoch()
                if val_accuracy > val_list[-1]:
                    val_list.append(val_accuracy)

                if self.loss_meter.value()[0] < best_loss:
                    logger.info('Found a better ckpt ({:.3f} -> {:.3f}), '.format(best_loss, self.loss_meter.value()[0]))
                    best_loss = self.loss_meter.value()[0]

                # visualize
                vis.plot('loss', self.loss_meter.value()[0])
                vis.plot('val_accuracy', val_accuracy)
                vis.log("epoch:{epoch},lr:{lr},loss:{loss},train_cm:{train_cm},val_cm:{val_cm}".format(
                    epoch=epoch, loss=self.loss_meter.value()[0], val_cm=str(val_cm.value()),
                    train_cm=str(self.confusion_matrix.value()), lr=get_learning_rates(self.optimizer)))

                for eval_brand_txt in self.val_data_brand_list.keys():
                    val_cm, val_accuracy = self._val_one_epoch(self.val_data_brand_list[eval_brand_txt])
                    val_name = 'val_accuracy'+eval_brand_txt
                    vis.plot(val_name, val_accuracy)
                    vis.log("epoch:{epoch},lr:{lr},val_cm:{val_cm}".format(
                       epoch=epoch, val_cm=str(val_cm.value()),lr=get_learning_rates(self.optimizer)))

            # adjust the lr
            #logger.info('lr is {}'.format(self.optimizer.get_lr()))
            if isinstance(self.lr_scheduler, ReduceLROnPlateau):
                self.lr_scheduler.step(self.loss_meter.value()[0], self.last_epoch)

#预训练的读取
    def _load_ckpt(self, ckpt):
        state_dict = t.load(ckpt)
        from collections import OrderedDict
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k[7:] # remove 'module.'
            new_state_dict[name] = v
        # load params
        self.model.load_state_dict(new_state_dict)

    def _train_one_epoch(self):
        for step, (data, label) in enumerate(self.train_data):
        #self.train_data_provider = DataIterator(self.train_data)
        #for i in range(0, 5):
            # data, label = self.train_data_provider.next()
            # train model
            inputs = Variable(data)
            target = Variable(label)
            if len(self.params.gpus) > 0:
                inputs = inputs.cuda()
                target = target.cuda()

            # forward
            score = self.model(inputs)
            loss = self.criterion(score, target)
	    #print(loss)
            # backward
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step(None)

            # meters update
            self.loss_meter.add(loss.item())
            #print("score is {}, target is {}".format(score.data, target.data))
            self.confusion_matrix.add(score.data, target.data)

    def _val_one_epoch(self, val_data):
        self.model.eval()
        confusion_matrix = meter.ConfusionMeter(self.num_classes)
        logger.info('Val on validation set...')

        for step, (data, label) in enumerate(val_data):

            # val model
            with t.no_grad():
                inputs = Variable(data)
                target = Variable(label.type(t.LongTensor))
                if len(self.params.gpus) > 0:
                    inputs = inputs.cuda()
                    target = target.cuda()

                score = self.model(inputs)
		#print(score)
                confusion_matrix.add(score.data.squeeze(), label.type(t.LongTensor))

        self.model.train()
        cm_value = confusion_matrix.value()
        print(cm_value)
        cm_value_correct = 0
        for i in range(0, self.num_classes):
            cm_value_correct += cm_value[i][i]
        accuracy = (100. * cm_value_correct ) / (cm_value.sum())
        print('accuracy is {}'.format(accuracy))
        return confusion_matrix, accuracy
