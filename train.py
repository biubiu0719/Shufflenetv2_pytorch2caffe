import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau

from data import DataSet
from utils import Trainer
#from network import cnn#, shufflenetv2
from shufflenetv2 import shufflenetv2
#from efficientnet_pytorch import EfficientNet
from shufflenetv2_test import shufflenetv2 as shuf
import sys
import argparse

# Hyper-params
data_root = './data/'
model_path = './models/'
# batch_size per GPU, if use GPU mode; resnet34: batch_size=120
batch_size = 1024
num_workers = 48
num_classes = 2
model_name = "shufflenet"
train_txt = ""
test_txt = ""
model_type = "s2"
model_width = 0.25

init_lr = 0.01
lr_decay = 0.8
momentum = 0.9
weight_decay = 0.000
nesterov = True

# Set Training parameters
params = Trainer.TrainParams()
params.max_epoch = 1000
params.criterion = nn.CrossEntropyLoss()
params.gpus = [2]  # set 'params.gpus=[]' to use CPU mode
params.save_dir = model_path
params.ckpt = None
params.save_freq_epoch = 2

parser = argparse.ArgumentParser()
parser.add_argument('--image_w', type=int, default=64)
parser.add_argument('--image_h', type=int, default=64)
parser.add_argument('--num_classes', type=int, default=2)
parser.add_argument('--batch_size', type=int, default=528)
parser.add_argument('--num_workers', type=int, default=24)
parser.add_argument('--model_name', type=str, default=None)
parser.add_argument('--train_txt', type=str, default=None)
parser.add_argument('--test_txt', type=str, default=None)
parser.add_argument('--train_mode', type=str, default='default')
parser.add_argument('--model_type', choices=["r34", "r101", "s1", "s2","ef","eff_caff","gfeff","s2_test"], default="s2")
parser.add_argument('--model_width', type=float, default=0.25)
parser.add_argument('--gpu', nargs='+', type=int)
parser.add_argument('--init_lr', type=float, default=0.01)
parser.add_argument('--lr_decay', type=float, default=0.8)
parser.add_argument('--momentum', type=float, default=0.9)
parser.add_argument('--weight_decay', type=float, default=0.000)
parser.add_argument('--max_epoch', type=int, default=1000)
parser.add_argument('--save_path', type=str, default=None)
parser.add_argument('--pretrained', type=str, default=None)
parser.add_argument('--save_freq_epoch', type=int, default=50)

parser.add_argument('--eval_each_brand', default=False, action='store_true')
parser.add_argument('--eval_brand_txt', type=str, default='data/val_dir1+data/val_dir2+data/val_dir3', help='path to all eval test dataset')

args = parser.parse_args()

assert(args.model_name is not None)
assert(args.train_txt is not None)
assert(args.test_txt is not None)
assert(args.save_path is not None)

batch_size = args.batch_size
num_workers = args.num_workers
num_classes = args.num_classes
model_name = args.model_name
train_txt = args.train_txt
test_txt = args.test_txt
model_type = args.model_type
model_width = args.model_width

init_lr = args.init_lr
lr_decay = args.lr_decay
momentum = args.momentum
weight_decay = args.weight_decay

params.max_epoch = args.max_epoch
params.gpus = args.gpu
params.save_dir = args.save_path
params.ckpt = args.pretrained
params.save_freq_epoch = args.save_freq_epoch
params.batch_size = params.batch_size
    

# load data
print("Loading dataset...")
train_data = DataSet(data_root,train=True,train_txt=train_txt,test_txt=test_txt,image_w=args.image_w,image_h=args.image_h,train_mode=args.train_mode)
val_data = DataSet(data_root,train=False,train_txt=train_txt,test_txt=test_txt,image_w=args.image_w,image_h=args.image_h,train_mode=args.train_mode)
val_data_brand_list = dict()
if args.eval_each_brand:
    eval_brand_txt_str = args.eval_brand_txt
    eval_brand_txt_list = eval_brand_txt_str.strip().split('+')
    for eval_brand_txt in eval_brand_txt_list:
        val_data_brand_list[eval_brand_txt] = DataSet(data_root,train=False,train_txt=train_txt,test_txt=eval_brand_txt,image_w=args.image_w,image_h=args.image_h,train_mode=args.train_mode)


batch_size = batch_size if len(params.gpus) == 0 else batch_size*len(params.gpus)

train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=num_workers)
print('train dataset len: {}'.format(len(train_dataloader.dataset)))

val_dataloader = DataLoader(val_data, batch_size=batch_size, shuffle=False, num_workers=num_workers)
print('val dataset len: {}'.format(len(val_dataloader.dataset)))

val_dataloader_brand_list = dict()
if args.eval_each_brand:
    for eval_brand_txt in val_data_brand_list.keys():
        val_dataloader_brand_list[eval_brand_txt] = DataLoader(val_data_brand_list[eval_brand_txt], batch_size=batch_size, shuffle=False, num_workers=num_workers)


if model_type == "s2":
    model = shufflenetv2(num_classes=num_classes, model_width=model_width, image_h=args.image_h, image_w=args.image_w)
#elif model_type == "cnn":
#    model = cnn.cnn(num_classes=num_classes, model_width=model_width, image_h=args.image_h, image_w=args.image_w)
#elif model_type == "ef":
#    model = efficientnet.efficientnet_b0(num_classes=num_classes)
#elif model_type == "eff_caff":
#    model = eff_caff.get_from_name("efficientnet-b0")
#elif model_type == "gfeff":
#    model = EfficientNet.from_name('efficientnet-b0')
#    feature = model._fc.in_features
#    model._fc = nn.Linear(in_features=feature,out_features=2,bias=True)
elif model_type=="s2_test":
    model = shuf(num_classes=num_classes, model_width=model_width, image_h=args.image_h, image_w=args.image_w) 
# optimizer
trainable_vars = [param for param in model.parameters() if param.requires_grad]
print("Training with sgd")
params.optimizer = torch.optim.SGD(trainable_vars, lr=init_lr,
                                   momentum=momentum,
                                   weight_decay=weight_decay,
                                   nesterov=nesterov)

# Train
params.lr_scheduler = ReduceLROnPlateau(params.optimizer, 'min', factor=lr_decay, patience=10, cooldown=10, verbose=True)
#print('init lr is {}'.format(params.optimizer.get_lr()))
trainer = Trainer(model, args.model_name,  params, train_dataloader, val_dataloader, val_dataloader_brand_list, train_data.get_data_len(), num_classes=args.num_classes)
trainer.train()
