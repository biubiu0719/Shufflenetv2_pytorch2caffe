#/usr/bin/env python
# encoding: utf-8

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
from PIL import Image
from torch.autograd import Variable
import torch.nn.functional as F
import torchvision.transforms.functional as tv_F

#import sys
#sys.path.append("~/home/xianbiao/python")
#sys.path.append("~/home/xianbiao/python/caffe")
#import os,sys,caffe

try:
    import caffe
    from caffe import layers as L
    from caffe import params as P
except ImportError:
    pass

import slim
from slim import g_name


class BasicBlock(nn.Module):

    def __init__(self, name, in_channels, out_channels, stride, dilation):
        super(BasicBlock, self).__init__()
        self.g_name = name
        self.in_channels = in_channels
        self.stride = stride
        channels = out_channels//2
        if stride == 1:
            assert in_channels == out_channels
            self.conv = nn.Sequential(
                slim.conv_bn_relu(name + '/conv1', channels, channels, 1),
                slim.conv_bn(name + '/conv2', 
                    channels, channels, 3, stride=stride, 
                    dilation=dilation, padding=dilation, groups=channels),
                slim.conv_bn_relu(name + '/conv3', channels, channels, 1),
            )
        else:
            self.conv = nn.Sequential(
                slim.conv_bn_relu(name + '/conv1', in_channels, channels, 1),
                slim.conv_bn(name + '/conv2', 
                    channels, channels, 3, stride=stride, 
                    dilation=dilation, padding=dilation, groups=channels),
                slim.conv_bn_relu(name + '/conv3', channels, channels, 1),
            )
            self.conv0 = nn.Sequential(
                slim.conv_bn(name + '/conv4', 
                    in_channels, in_channels, 3, stride=stride, 
                    dilation=dilation, padding=dilation, groups=in_channels),
                slim.conv_bn_relu(name + '/conv5', in_channels, channels, 1),
            )
        self.shuffle = slim.channel_shuffle(name + '/shuffle', 2)

    def forward(self, x):
        if self.stride == 1:
            x1 = x[:, :(x.shape[1]//2), :, :]
            x2 = x[:, (x.shape[1]//2):, :, :]
            x = torch.cat((x1, self.conv(x2)), 1)
        else:
            x = torch.cat((self.conv0(x), self.conv(x)), 1)
        return self.shuffle(x)

    def generate_caffe_prototxt(self, caffe_net, layer):
        if self.stride == 1:
            layer_x1, layer_x2 = L.Slice(layer, ntop=2, axis=1, slice_point=[self.in_channels//2])
            caffe_net[self.g_name + '/slice1'] = layer_x1
            caffe_net[self.g_name + '/slice2'] = layer_x2
            layer_x2 = slim.generate_caffe_prototxt(self.conv, caffe_net, layer_x2)
        else:
            layer_x1 = slim.generate_caffe_prototxt(self.conv0, caffe_net, layer)
            layer_x2 = slim.generate_caffe_prototxt(self.conv, caffe_net, layer)
        layer = L.Concat(layer_x1, layer_x2, axis=1)
        caffe_net[self.g_name + '/concat'] = layer
        layer = slim.generate_caffe_prototxt(self.shuffle, caffe_net, layer)
        return layer


class Network(nn.Module):

    def __init__(self, num_classes, width_multiplier, image_h, image_w):
        super(Network, self).__init__()
        width_config = {
            0.02:(6,12,24,128),
            0.04:(12,24,48,256),
            0.05:(12,24,48,512),
            0.1:(24,48,96,512),
            0.125:(12,24,48,96,1024),
            0.25: (24, 48, 96, 192, 1024),
            0.041:(12,24,48,48,256),
            0.031:(8,16,32,32,256),
            0.021:(6,12,24,24,128),
            0.022:(6,12,24,24,128),
            0.015:(4,8,16,16,96),
        }
        width_config = width_config[width_multiplier]
        self.num_classes = num_classes
        #in_channels = 24
        in_channels=width_config[0]
        # outputs, stride, dilation, blocks, type
        self.network_config = [
            g_name('data/bn', nn.BatchNorm2d(3)),
            slim.conv_bn_relu('stage1/conv', 3, in_channels, 3, 2, 1),
            # g_name('stage1/pool', nn.MaxPool2d(3, 2, 1)),
            g_name('stage1/pool', nn.MaxPool2d(3, 2, 0, ceil_mode=True)),
            (width_config[1], 2, 1, 2, 'b'),
            (width_config[2], 2, 1, 2, 'b'), # x16
            (width_config[3], 2, 1, 2, 'b'), # x32
            slim.conv_bn_relu('conv5', width_config[3], width_config[4], 1),
            #g_name('pool', nn.AvgPool2d(7)),
            #g_name('pool', nn.AvgPool2d((2, 2))),
            #g_name('fc', nn.Conv2d(width_config[3], self.num_classes, (int(image_h / 32), int(image_w / 32)))),
            g_name('fc', nn.Conv2d(width_config[4], self.num_classes, 1))
            #slim.flatten('flatten', 1),
            #g_name('fc', nn.Linear(width_config[4], self.num_classes, bias=False))
        ]
        self.network = []
        for i, config in enumerate(self.network_config):
            if isinstance(config, nn.Module):
                self.network.append(config)
                continue
            out_channels, stride, dilation, num_blocks, stage_type = config
            stage_prefix = 'stage_{}'.format(i - 1)
            blocks = [BasicBlock(stage_prefix + '_1', in_channels, 
                out_channels, stride, dilation)]
            for i in range(1, num_blocks):
                blocks.append(BasicBlock(stage_prefix + '_{}'.format(i + 1), 
                    out_channels, out_channels, 1, dilation))
            self.network += [nn.Sequential(*blocks)]

            in_channels = out_channels
        self.network = nn.Sequential(*self.network)

        for name, m in self.named_modules():
            if any(map(lambda x: isinstance(m, x), [nn.Linear, nn.Conv1d, nn.Conv2d])):
                nn.init.kaiming_uniform_(m.weight, mode='fan_in')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def trainable_parameters(self):
        parameters = [
            {'params': self.cls_head_list.parameters(), 'lr_mult': 1.0},
            {'params': self.loc_head_list.parameters(), 'lr_mult': 1.0},
            # {'params': self.network.parameters(), 'lr_mult': 0.1},
        ]
        for i in range(len(self.network)):
            lr_mult = 0.1 if i in (0, 1, 2, 3, 4, 5) else 1
            parameters.append(
                {'params': self.network[i].parameters(), 'lr_mult': lr_mult}
            )
        return parameters

    def forward(self, x):
        x = self.network(x)
        return x.reshape(x.shape[0], -1)

    def generate_caffe_prototxt(self, caffe_net, layer):
        data_layer = layer
        network = slim.generate_caffe_prototxt(self.network, caffe_net, data_layer)
        return network

    def convert_to_caffe(self, name):
        caffe_net = caffe.NetSpec()
        layer = L.Input(shape=dict(dim=[1, 3, args.image_h, args.image_w]))
        caffe_net.tops['data'] = layer
        slim.generate_caffe_prototxt(self, caffe_net, layer)
        print(caffe_net.to_proto())
        with open(name + '.prototxt', 'wb') as f:
            f.write(str(caffe_net.to_proto()).encode())
        #caffe.set_mode_gpu()
        #caffe.set_device(2)
        caffe_net = caffe.Net(name + '.prototxt', caffe.TEST)
        slim.convert_pytorch_to_caffe(self, caffe_net)
        caffe_net.save(name + '.caffemodel')

def shufflenetv2(num_classes, model_width, image_h, image_w):
    net = Network(num_classes, model_width, image_h, image_w)
    return net

def _load_ckpt(model, ckpt):
        state_dict = torch.load(ckpt)
        from collections import OrderedDict
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k[7:] # remove 'module.'
            new_state_dict[name] = v
        # load params
        model.load_state_dict(new_state_dict,False)

if __name__ == '__main__':
# main_main():
    import sys
    import argparse
    import PIL.Image
    import torchvision
    import numpy as np
    import os
    os.environ['CUDA_VISIBLE_DEVICES'] = '2,3'

    def assert_diff(a, b):
        print('pytorch result is {}, caffe result is {}'.format(a, b))
        if isinstance(a, torch.Tensor):
            a = a.detach().cpu().numpy()
        if isinstance(b, torch.Tensor):
            b = b.detach().cpu().numpy()
        print(a.shape, b.shape)
        a = a.reshape(-1)
        b = b.reshape(-1)
        assert a.shape == b.shape
        diff = np.abs(a - b)
        print('mean diff = %f' % diff.mean())
        assert diff.mean() < 0.001
        print('max diff = %f' % diff.max())
        assert diff.max() < 0.001

    parser = argparse.ArgumentParser()
    parser.add_argument('--image_h', type=int, default=128)
    parser.add_argument('--image_w', type=int, default=64)
    parser.add_argument('--num_classes', type=int, default=3)
    parser.add_argument('--model_width', type=float, default=0.25)
    parser.add_argument('--load_pytorch', type=str)
    parser.add_argument('--save_pytorch', type=str)
    parser.add_argument('--save_caffe', type=str)
    parser.add_argument('--test', type=str)
    args = parser.parse_args()

#    if args.test is None:
    img = np.random.rand(1, 3, args.image_h, args.image_w)
        # img = np.ones((1, 3, args.image_hw, args.image_hw))
#    else:
#        img = PIL.Image.open(args.test).convert('RGB')
#        img = torchvision.transforms.functional.resize(img, (args.image_h, args.image_w))
#        img = tv_F.normalize(tv_F.to_tensor(img), [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
#        img = (img).unsqueeze(0).numpy()
        #img = Image.open(args.test)
        #img = tv_F.to_tensor(tv_F.resize(img, (64, 64)))
        #img = tv_F.normalize(img, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        #img_input = Variable(torch.unsqueeze(img, 0))


    ##############################################
    # Initilize a PyTorch model.
    net = Network(args.num_classes, args.model_width, args.image_h, args.image_w).train(False)
    print(net)
    if args.load_pytorch is not None:
        #net.load_state_dict(torch.load(args.load_pytorch, map_location=lambda storage, loc: storage))
        _load_ckpt(net, args.load_pytorch)
    x = torch.tensor(img.copy(), dtype=torch.float32)
    with torch.no_grad():
        cls_results = net(x)
    print(cls_results.shape)
    if args.save_pytorch is not None:
        torch.save(net.state_dict(), args.save_pytorch + '.pth')

    ##############################################
    # Caffe model generation and converting.
    if args.save_caffe is not None:
        net.convert_to_caffe(args.save_caffe)
        caffe_net = caffe.Net(args.save_caffe + '.prototxt', caffe.TEST, 
            weights=(args.save_caffe + '.caffemodel'))
        caffe_net.blobs['data'].data[...] = img.copy()
        caffe_results = caffe_net.forward(blobs=['fc'])
        cls_results_caffe = caffe_results['fc']
        print(cls_results_caffe.shape)
        assert_diff(cls_results, cls_results_caffe)
