import os
a=0
#1024为我们的类别数
dir1 = './../../data/20190704/positive/'#图片文件的地址
dir2='./../../data/20190704/negative_true/'
dir3= './../../data/20190711/positive/'#图片文件的地址
dir4='./../../data/20190711/negative_true/'
dir5= './../../../data/20190715/positive/'#图片文件的地址
dir6='./../../../data/20190715/negative_true/'
dir7='./../../../data/20200303/positive/'
dir8='./../../../data/20200303/negative/'
dir9='./../../data/20200325/'
label1=1
label0=0
#os.listdir的结果就是一个list集，可以使用list的sort方法来排序。如果文件名中有数字，就用数字的排序
train = open('./nuw.txt','a')
text = open('./check.txt', 'a')
#前三个文档
# for num in range(0,3150):
#     name =  str(dir1) +  str(num)+'.jpg' + ' ' + str(int(label0)) +'\n'
#     train.write(name)

# for num in range(0,6300):
#     name =  str(dir2) +  str(num)+'.jpg' + ' ' + str(int(label1)) +'\n'
#     train.write(name)


# for num in range(0,2800):
#     name =  str(dir3) +  str(num)+'.jpg' + ' ' + str(int(label0)) +'\n'
#     train.write(name)   

# for num in range(0,5700):
#     name =  str(dir4) +  str(num)+'.jpg' + ' ' + str(int(label1)) +'\n'
#     train.write(name)   


# for num in range(0,12000):
#     name =  str(dir5) +  str(num)+'.jpg' + ' ' + str(int(label0)) +'\n'
#     train.write(name)   

# for num in range(0,28000):
#     name =  str(dir6) +  str(num)+'.jpg' + ' ' + str(int(label1)) +'\n'
#     train.write(name)   

# for num in range(0,19000):
#     name =  str(dir7) +  str(num)+'.jpg' + ' ' + str(int(label0)) +'\n'
#     train.write(name)   

# for num in range(0,39000):
#     name =  str(dir8) +  str(num)+'.jpg' + ' ' + str(int(label1)) +'\n'
#     train.write(name)   



for num in range(12000,17200):
    name =  str(dir5) +  str(num)+'.jpg' + ' ' + str(int(label0)) +'\n'
    text.write(name)   

for num in range(28000,34400):
    name =  str(dir6) +  str(num)+'.jpg' + ' ' + str(int(label1)) +'\n'
    text.write(name)   

for num in range(19000,24700):
    name =  str(dir7) +  str(num)+'.jpg' + ' ' + str(int(label0)) +'\n'
    text.write(name)   

for num in range(39000,49000):
    name =  str(dir8) +  str(num)+'.jpg' + ' ' + str(int(label1)) +'\n'
    text.write(name)   
# for num in range(0,4260):
#     name =  str(dir9) +  str(num)+'.jpg' + ' ' + str(int(label0)) +'\n'
#     train.write(name)   
text.close()
train.close()
