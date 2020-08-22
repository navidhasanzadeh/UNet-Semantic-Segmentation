# -*- coding: utf-8 -*-
"""
Created on Tue Mar 26 15:33:29 2019

@author: Navid
"""

import glob
import cv2
import numpy as np
import matplotlib.pyplot as plt
from collections import OrderedDict
import os
import sys
class PrepareCamVid:
    def __init__(self,images_path, labels_path, save_path, resize=False):
        images_path = images_path
        self.images = sorted(glob.glob(images_path + "*.png"))
        labels_path = labels_path
        self.labels = sorted(glob.glob(labels_path + "*.png"))
        self.save_path =save_path
        self.resize = resize
        self.ade20k = dic = {0:127,1:2,2:13,3:62,4:2, 5:21, 6:0, 7:13, 8:43, 9:33, 10:7, 11:12, 12:0 ,13:117, 14:0, 15:0, 16:13,17:7,18:7,19:12,20:44,21:3,22:84,23:0,24:137,25:81,26:5,27:84,28:53,29:10,30:1,31:0}
        camvid_colors = OrderedDict([
            ("Animal", np.array([64, 128, 64], dtype=np.uint8)),
            ("Archway", np.array([192, 0, 128], dtype=np.uint8)),
            ("Bicyclist", np.array([0, 128, 192], dtype=np.uint8)),
            ("Bridge", np.array([0, 128, 64], dtype=np.uint8)),
            ("Building", np.array([128, 0, 0], dtype=np.uint8)),
            ("Car", np.array([64, 0, 128], dtype=np.uint8)),
            ("CartLuggagePram", np.array([64, 0, 192], dtype=np.uint8)),
            ("Child", np.array([192, 128, 64], dtype=np.uint8)),
            ("Column_Pole", np.array([192, 192, 128], dtype=np.uint8)),
            ("Fence", np.array([64, 64, 128], dtype=np.uint8)),
            ("LaneMkgsDriv", np.array([128, 0, 192], dtype=np.uint8)),
            ("LaneMkgsNonDriv", np.array([192, 0, 64], dtype=np.uint8)),
            ("Misc_Text", np.array([128, 128, 64], dtype=np.uint8)),
            ("MotorcycleScooter", np.array([192, 0, 192], dtype=np.uint8)),
            ("OtherMoving", np.array([128, 64, 64], dtype=np.uint8)),
            ("ParkingBlock", np.array([64, 192, 128], dtype=np.uint8)),
            ("Pedestrian", np.array([64, 64, 0], dtype=np.uint8)),
            ("Road", np.array([128, 64, 128], dtype=np.uint8)),
            ("RoadShoulder", np.array([128, 128, 192], dtype=np.uint8)),
            ("Sidewalk", np.array([0, 0, 192], dtype=np.uint8)),
            ("SignSymbol", np.array([192, 128, 128], dtype=np.uint8)),
            ("Sky", np.array([128, 128, 128], dtype=np.uint8)),
            ("SUVPickupTruck", np.array([64, 128, 192], dtype=np.uint8)),
            ("TrafficCone", np.array([0, 0, 64], dtype=np.uint8)),
            ("TrafficLight", np.array([0, 64, 64], dtype=np.uint8)),
            ("Train", np.array([192, 64, 128], dtype=np.uint8)),
            ("Tree", np.array([128, 128, 0], dtype=np.uint8)),
            ("Truck_Bus", np.array([192, 128, 192], dtype=np.uint8)),
            ("Tunnel", np.array([64, 0, 64], dtype=np.uint8)),
            ("VegetationMisc", np.array([192, 192, 0], dtype=np.uint8)),
            ("Wall", np.array([64, 192, 0], dtype=np.uint8)),
            ("Void", np.array([0, 0, 0], dtype=np.uint8))
        ])
            
        camvid_colors_list = list(camvid_colors.values())
        self.camvid_colors_list = [c.tolist() for c in camvid_colors_list]
        
    def prepare_labels(self,ade20k=False):
        print('')
        print('Labels Preparation')
        if not os.path.exists(self.save_path + r"prepared_labels/"):
            os.makedirs(self.save_path + r"prepared_labels/")
        for i,img in enumerate(self.labels):
            if not os.path.exists(self.save_path + r"prepared_labels/" + str(i) + r'.png'):            
#                import pdb; pdb.set_trace()
                label = cv2.imread(img)
                label = cv2.cvtColor(label, cv2.COLOR_BGR2RGB)
                label_gray = np.zeros([label.shape[0],label.shape[1]])
                if not ade20k:
                    for m in range(label.shape[0]):
                        for n in range(label.shape[1]):
                            try:
                                label_gray[m,n] = self.camvid_colors_list.index(label[m,n,:].tolist())
                            except:
                                label_gray[m,n] = 31
                else:
                    for m in range(label.shape[0]):
                        for n in range(label.shape[1]):
                            try:
                                label_gray[m,n] = self.ade20k[self.camvid_colors_list.index(label[m,n,:].tolist())]
                            except:
                                label_gray[m,n] = 0
                    
                if self.resize is not False:
                    label_gray =  cv2.resize(label_gray, self.resize)
                cv2.imwrite(self.save_path + r"prepared_labels/" + str(i) + r'.png', label_gray)
            self.__update_progress((i+1)/len(self.labels))
        return self.save_path + r"prepared_labels/"
    def prepare_images(self,flip=True):
        print('')
        print('Images Preparation')
        if not os.path.exists(self.save_path + r"prepared_images/"):
            os.makedirs(self.save_path + r"prepared_images/")
#        import pdb; pdb.set_trace()
        for i,img in enumerate(self.images):            
            if not os.path.exists(self.save_path + r"prepared_images/" + str(i) + r'.png'):
                image = cv2.imread(img)
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                image = cv2.resize(image, self.resize)
                if flip is True:
                    big_image = np.zeros([572,572,3])        
                    big_image = big_image.astype(np.uint8)                
                    big_image[92:572-92,92:572-92,:] = image
                    big_image[0:92,92:572-92,:] = image[92:0:-1,:,:]
                    big_image[572-92::1,92:572-92,:] = image[387:387-92:-1,:,:]
                    big_image[92:572-92,0:92,:] = image[:,92:0:-1,:]
                    big_image[92:572-92,572-92::1,:] = image[:,387:387-92:-1,:]
                    big_image[0:92,0:92,:] = image[92:0:-1,92:0:-1,:]
                    big_image[572-92:572,572-92:572,:] = image[387:387-92:-1,387:387-92:-1,:]
                    big_image[572-92:572:1,92:0:-1,:] = image[387:387-92:-1,0:92:1,:]
                    big_image[92:0:-1,572-92:572:1,:] = image[0:92:1,387:387-92:-1,:]        
                else:
                    big_image = image
                    big_image = big_image.astype(np.uint8) 
                cv2.imwrite(self.save_path + r"prepared_images/" + str(i) + r'.png', cv2.cvtColor(big_image, cv2.COLOR_RGB2BGR))
            self.__update_progress((i+1)/len(self.images))
        return self.save_path + r"prepared_images/"
    def __update_progress(self,progress):
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
        text = "\rPercent: [{0}] {1:.2f}% {2}".format( "#"*block + "-"*(barLength-block), progress*100, status)
        sys.stdout.write(text)
        sys.stdout.flush()                    