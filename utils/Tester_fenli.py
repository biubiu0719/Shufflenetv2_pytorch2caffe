from __future__ import print_function

import os
from PIL import Image
from .log import logger

import torch
from torch.autograd import Variable
import torch.nn.functional as F
import torchvision.transforms.functional as tv_F


class TestParams(object):
    # params based on your local env
    gpus = []  # default to use CPU mode

    # loading existing checkpoint
    ckpt = './models/shufflenetv2_qeelin/ckpt_epoch_999.pth'     # path to the ckpt file

    testdata_dir = './testimg/'

class Tester(object):

    TestParams = TestParams

    def __init__(self, model, test_params, img_txt, output_txt, image_w, image_h):
        assert isinstance(test_params, TestParams)
        self.params = test_params
        self.img_txt = img_txt
        self.output_txt = output_txt
        self.image_w = image_w
        self.image_h = image_h

        # load model
        self.model = model
        ckpt = self.params.ckpt
        if ckpt is not None:
            self._load_ckpt(ckpt)
            logger.info('Load ckpt from {}'.format(ckpt))

        # set CUDA_VISIBLE_DEVICES, 1 GPU is enough
        if len(self.params.gpus) > 0:
            gpu_test = str(self.params.gpus[0])
            os.environ['CUDA_VISIBLE_DEVICES'] = gpu_test
            logger.info('Set CUDA_VISIBLE_DEVICES to {}...'.format(gpu_test))
            self.model = self.model.cuda()

        self.model.eval()

    def test(self):

        #img_txt = './images/test_staff_qeelin.txt'
        #output_txt = './predictResult/test_staff_qeelin_predict_shuffv2.txt'
        img_txt = self.img_txt#'./demo_staff/202018000956_test.txt'
        output_txt = self.output_txt#'./demo_staff/202018000956_test_pred.txt'
        total_pred = []
        img_index = 0
        lable_check=0
        curreckcount=0
        coor=0
        numb=0
        with open(img_txt, 'r') as file:
            for line in file.readlines():
                imgPath, label = line.strip().split(' ')
        #img_list = os.listdir(self.params.testdata_dir)

        #for img_name in img_list:
                #print('Processing image: ' + imgPath)
                #check
                
                path0='./../../data/0325pos4/'
                path1='./../../data/0325neg4/'
                img2= Image.open(imgPath)
                img = Image.open(imgPath)
                #check
                img = tv_F.to_tensor(tv_F.resize(img, (self.image_h, self.image_w)))
                img = tv_F.normalize(img, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                img_input = Variable(torch.unsqueeze(img, 0))
                if len(self.params.gpus) > 0:
                    img_input = img_input.cuda()

                output = self.model(img_input)
		#print(output)
                score = F.softmax(output, dim=1)
                #print("score is {}, shape is {}\n".format(score, score.shape))
                str_s = ''
                save_path=''
                for j in range(0, score.shape[1]):
                    s = (score[0][j].float())
                    str_s += ' ' + '{}'.format(s)
                if(score[0][0].float()>=score[0][1].float()):
                    save_path=path0+'{}'.format(numb)+'.jpg'
                    img2.save(save_path)
                    lable_check=0
                else:
                    save_path=path1+'{}'.format(numb)+'.jpg'
                    img2.save(save_path)
                    lable_check=1
                if(lable_check==int(label)):
                    curreckcount+=1
                numb+=1
		#print(curreckcount)
                #s = (score[0][1].float())
                #print('s is {}'.format(s))
                #str_s = str(s)
                #str_s = '{}'.format(s)
                #print('{}: s_str is {}'.format(img_index, str_s))
                img_index += 1
                line = imgPath + ' ' + label + str_s  + '\n'
                total_pred.append(line)
                #print('Score is {}'.format(score[0][1]))
                _, prediction = torch.max(score.data, dim=1)
                if(prediction[0]==int(label)):
                    coor+=1

                #print('Prediction number: ' + str(prediction[0]))
        file.close()
        print(curreckcount)
        print(img_index)
        print(coor)
        accu=100*curreckcount/img_index
        print(accu)
        with open(output_txt, 'w') as fileWrite:
            fileWrite.writelines(total_pred)
        fileWrite.close()

    def _load_ckpt(self, ckpt):
        #self.model.load_state_dict(torch.load(ckpt))
        state_dict = torch.load(ckpt)
        from collections import OrderedDict
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k[7:] # remove 'module.'
            new_state_dict[name] = v
        # load params
        self.model.load_state_dict(new_state_dict)
