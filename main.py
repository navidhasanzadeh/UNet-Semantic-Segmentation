# -*- coding: utf-8 -*-
"""
Created on Sat Mar 23 22:02:13 2019

@author: Navid
"""
#from IPython.display import clear_output

import torch
import torch.nn as nn
import torchvision.transforms as transform
import torchvision.datasets as dsets
from torch.autograd import Variable
import glob
import sys
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from data_preparation import PrepareCamVid as pc
from unet_model import UNetModel
#%matplotlib inline

def __update_progress(epoch,i,progress,loss):
    barLength = 10 # Modify this to change the length of the progress bar
    status = ""
    if isinstance(progress, int):
        progress = float(progress)
    if not isinstance(progress, float):
        progress = 0
        status = "error: progress var must be float\r\n"
    if progress < 0:
        progress = 0
        status = "Halt...\r\n"
    if progress >= 1:
        progress = 1
        status = "Done...\r\n"
    block = int(round(barLength*progress))
    text = "\rPercent: [{0}] epoch={4} i={5} {1:.2f}% loss={2:.2f} {3}".format( "#"*block + "-"*(barLength-block), progress*100,loss, status,epoch,i)
    sys.stdout.write(text)
    sys.stdout.flush()                    
    
width = 388
height = 388

images_path = r'./camvid/701_StillsRaw_full/'
labels_path =  r'./camvid/LabeledApproved_full/'
save_path = r'./data/camvid-processed/'

data_prepare = pc(images_path=images_path,labels_path =labels_path, save_path=save_path,resize=(width,height))
saved_labels_path = data_prepare.prepare_labels()
saved_images_path = data_prepare.prepare_images()


images = sorted(glob.glob(saved_images_path + "*.png"),key=lambda i: int(os.path.splitext(os.path.basename(i))[0]))
labels = sorted(glob.glob(saved_labels_path + "*.png"),key=lambda i: int(os.path.splitext(os.path.basename(i))[0]))

train_loader = [(images[i], labels[i]) for i in range(0,516,1)]
test_loader = [(images[i], labels[i]) for i in range(516,len(images),1)]

batch_size = 1
n_iters = 3000 # how many times weights are updated
num_epochs = n_iters/(len(images)/batch_size)
num_epochs = int(num_epochs)


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        torch.nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.01)
model = UNetModel()
model.apply(weights_init)

if torch.cuda.is_available():
    model.cuda()
if torch.cuda.is_available():
    criterion = nn.CrossEntropyLoss().cuda() #softmax + crossEntropy
else:
    criterion = nn.CrossEntropyLoss()

learning_rate = 0.0001
num_epochs = 40
model_id = 1

train_losses = []
test_losses = []
if not os.path.exists('./data2/test_images/{}/'.format(model_id)):
    os.makedirs('./data2/test_images/{}/'.format(model_id))
if not os.path.exists('./data2/models/{}/'.format(model_id)):
    os.makedirs('./data2/models/{}/'.format(model_id))
for epoch in range(num_epochs):
#    clear_output()
    learning_rate = learning_rate * 0.995
    optimizer = torch.optim.Adam(model.parameters(),lr=learning_rate, weight_decay=1e-4)
    train_epoch_loss = []
    print(':::Train:::')
    for i, (img,mask) in enumerate(train_loader):
        images = cv2.imread(img)      
        images = images.astype(np.float32)
        for j in range(3):
            images[:,:,j] = images[:,:,j] / np.max(images[:,:,j])        
        images = torch.from_numpy(np.array([[images[:,:,0],images[:,:,1],images[:,:,2]]]))
        labels = torch.from_numpy(np.array([cv2.imread(mask,0)]))
        if torch.cuda.is_available():
            images = Variable(images.cuda())
            labels = Variable(labels.cuda())           
        else:
            images = Variable(images)
            labels = Variable(labels)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs.to(dtype=torch.float32).view(1,32,388,-1),torch.max(labels,dim=0)[0].to(dtype=torch.int64).view(1,388,388))
        train_epoch_loss.append(float(loss))  
        loss.backward()
        optimizer.step()
        __update_progress(epoch,i,(i+1)/len(train_loader),train_epoch_loss[-1])
        del loss
#        print('train::epoch={} , i={}, loss={}'.format(epoch,i,train_epoch_loss[-1]))
    train_losses.append(np.mean(train_epoch_loss))
    if True:
        if not os.path.exists('./data2/test_images/{}/{}/'.format(model_id,epoch)):
          os.makedirs('./data2/test_images/{}/{}/'.format(model_id,epoch))
        print(':::Test:::')
        test_epoch_loss = []
        for i, (img,mask) in enumerate(test_loader):
            images = cv2.imread(img)      
            images = images.astype(np.float32)
            for j in range(3):
                images[:,:,j] = images[:,:,j] / np.max(images[:,:,j])        
            images = torch.from_numpy(np.array([[images[:,:,0],images[:,:,1],images[:,:,2]]]))
            labels = torch.from_numpy(np.array([cv2.imread(mask,0)]))
            if torch.cuda.is_available():
                images = Variable(images.cuda())
                labels = Variable(labels.cuda())           
            else:
                images = Variable(images)
                labels = Variable(labels)                        
            outputs = model(images)
            loss = criterion(outputs.to(dtype=torch.float32).view(1,32,388,-1),torch.max(labels,dim=0)[0].to(dtype=torch.int64).view(1,388,388))
            test_epoch_loss.append(float(loss))                          
            o_est= torch.max(outputs[0],dim=0)[1].cpu().numpy()
            o_real=torch.max(labels,dim=0)[0].cpu().numpy()   
            __update_progress(epoch,i,(i+1)/len(test_loader),test_epoch_loss[-1])
            del loss
            if(i%int(len(test_loader)/4)==0):
#                plt.figure()
#                plt.imshow(cv2.normalize(images.cpu().numpy()[0][0][92:572-92,92:572-92], None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F),cmap='gray')
#                plt.show()
#                plt.figure()
#                plt.imshow(cv2.normalize(o_real, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F),cmap='bwr')
#                plt.show()
#                plt.figure()
#                plt.imshow(cv2.normalize(o_est, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F),cmap='bwr')
#                plt.show()            
                plt.imsave(r'./data2/test_images/{}/{}/{}-{}-main.png'.format(model_id,epoch,epoch,i),np.uint8(images.cpu().numpy()[0][0][92:572-92,92:572-92]*255.9))
                plt.imsave('./data2/test_images/{}/{}/{}-{}-target.png'.format(model_id,epoch,epoch,i),np.uint8(255.9*cv2.normalize(o_real, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)),cmap='jet')
                plt.imsave('./data2/test_images/{}/{}/{}-{}-estimated.png'.format(model_id,epoch,epoch,i),np.uint8(255.9*cv2.normalize(o_est, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)),cmap='jet')
        test_losses.append(np.mean(test_epoch_loss))
#    torch.save(model.state_dict(), './data2/models/{}/unet-epoch{}.pkl'.format(model_id,epoch))
import pickle
with open('./data2/models/{}/vars.pkl'.format(model_id), 'wb') as f:
    pickle.dump([model, train_losses, test_losses], f)

import matplotlib.pyplot as plt
fig=plt.figure()
plt.plot(train_losses)
plt.ylabel('train loss')
plt.xlabel('epoch')
fig.savefig('./data2/models/{}/train_loss.png'.format(model_id))
fig=plt.figure()
plt.plot(test_losses)
plt.ylabel('test loss')
plt.xlabel('epoch')
fig.savefig('./data2/models/{}/test_loss.png'.format(model_id))

pytorch_total_params = sum(p.numel() for p in model.parameters())
print('Total Params = {}'.format(pytorch_total_params))
#!tar -cvf ./data2.tar ./data2
#from google.colab import files
#files.download("./data2.tar")
