#!/usr/bin/env python
# coding: utf-8

import torch
from torch import nn
from models.layers import CVAE
from task.loss import HeatmapLoss
import torch.nn.functional as F
import torchvision
from torch.utils.data import DataLoader, TensorDataset
import utils.utils
import matplotlib.pyplot as plt
import os
import numpy as np
import cv2
from models.shnnet import Stacked_hourglass


images = []
landmarks = []

path = r'D:\jyn\thesis\code\dataset\AFLW'
filenames = os.listdir(path)
for filename in filenames:
    images.append(cv2.imread(os.path.join(path, filename),0))
images = np.array(images).reshape(2995,-1)
    
with open(r"D:\jyn\thesis\code\dataset\AFLW\testing.txt", "r",encoding="utf-8") as f:  # 
    data = f.read()  
landmark = data.replace('\n','').split(" ")
landmark = np.array(landmark).reshape(2995,15)
landmarks = landmark[:,1:-4].astype(np.float64).reshape(2995,10)

x_train = images.astype("float32")/255
y_train = landmarks.astype("float32")

# loading data
train_data = TensorDataset(torch.tensor(x_train), torch.tensor(y_train))


if __name__ == '__main__':
    epochs = 100
    batch_size = 100

    recon = None
    img = None

    #utils.make_dir("C:/Users/jyn/cvae_shn_model/weights.pth")

    data_loader = DataLoader(train_data, batch_size=100, shuffle=True)

    #cvae = CVAE(image_size=128, label_size=136, latent_size=150)
    cvae = CVAE(image_size=150, label_size=136, latent_size=150)
    shn = Stacked_hourglass(nstack, inp_dim, oup_dim, bn=False, increase=0, **kwargs)

    optimizer = torch.optim.Adam(cvae.parameters(), lr=1e-3)

    epochs = 10

    for epoch in range(epochs):
        train_loss = 0
        i = 0
        for batch_id, data in enumerate(data_loader):
            inputs, y = data
            y_hat = shn(inputs)
            loss_1 = shn.calc_loss(y_hat,y)
            loss_1.backward()
            optimizer.step()
            for batch_id, data in enumerate(data_loader):
                x, Y_ = data
                Y = shn(x)
                loss_4 = shn.calc_loss(Y,Y_)
                loss_4.backward()
                optimizer.step()
                
                x_gen, mu, log_std = cvae(x, y_hat)
                y_ = shn(x_gen)
                loss_2 = shn.calc_loss(y_,y_hat)
                optimizer.zero_grad()
                loss_2.backward()
                optimizer.step()
                
                x_rec, mu, log_std = cvae(x_gen, Y)
                loss_3 = cvae.loss_function(x_rec, x, mu, log_std)
                optimizer.zero_grad()
                loss_3.backward()
                optimizer.step()

            train_loss += loss.item()
            i += 1

            #if batch_id % 10 == 0:
               #print("Epoch[{}/{}], Batch[{}/{}], batch_loss:{:.6f}".format(
                    #epoch+1, epochs, batch_id+1, len(data_loader), loss.item()))

        print("======>epoch:{},\t epoch_average_batch_loss:{:.6f}============".format(epoch+1, train_loss/i), "\n")

        # save imgs
        #if epoch % 10 == 0:
            #imgs = utils.to_img(recon.detach())
          #  path = "./img/cvae_shn_model/epoch{}.png".format(epoch+1)
            #torchvision.utils.save_image(imgs, path, nrow=10)
           # print("save:", path, "\n")

    # save val model
    #utils.save_model(cvae, "./model_weights/cvae_shn_model/weights.pth")