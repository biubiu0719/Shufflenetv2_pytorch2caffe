from torch import nn
from utils import Tester
from shufflenetv2 import shufflenetv2
#from network import efficientnet
from shufflenetv2_test import shufflenetv2 as shuf
import sys
import argparse
import time
import decimal
#from shufflentv2_test import shufflenetv2 as shuf
parser = argparse.ArgumentParser()
parser.add_argument('--image_w', type=int, default=64)
parser.add_argument('--image_h', type=int, default=64)
parser.add_argument('--num_classes', type=int, default=2)
parser.add_argument('--test_txt', type=str, default=None)
parser.add_argument('--model_type', choices=["r34", "r101", "s1", "s2","ef","s2_test"], default="s2")
parser.add_argument('--ckpt', type=str, default=None)
parser.add_argument('--output_txt', type=str, default=None)
parser.add_argument('--model_width', type=float, default=0.25)
parser.add_argument('--gpu', nargs='+', type=int)
args = parser.parse_args()

assert(args.ckpt is not None)
assert(args.output_txt is not None)

# Set Test parameters
params = Tester.TestParams()
params.gpus = args.gpu  # set 'params.gpus=[]' to use CPU model. if len(params.gpus)>1, default to use params.gpus[0] to test
params.ckpt = args.ckpt  #'./models/ckpt_epoch_400_res34.pth'

model_path = './models/'

# models

if args.model_type == "s2":
	model = shufflenetv2(num_classes=args.num_classes, model_width=args.model_width, image_h=args.image_h, image_w=args.image_w)
elif args.model_type == "s2_test":
    model = shuf(num_classes=args.num_classes, model_width=args.model_width, image_h=args.image_h, image_w=args.image_w)

# Test
tester = Tester(model, params, args.test_txt, args.output_txt, args.image_w, args.image_h)
start=time.time()
tester.test()
end=time.time()
print("running time:{} seconds".format(end-start))
