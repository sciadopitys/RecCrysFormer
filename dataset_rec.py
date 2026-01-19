from torch.utils.data import Dataset, DataLoader

import torch 
import torch.nn as nn
import torch.fft
import torch.cuda
import numpy as np
from numpy import sqrt

import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

import time 

# rounding to nearest even number
def round2( n ): 
  
    # Smaller multiple 
    a = round(n // 2) * 2
      
    # Larger multiple 
    b = a + 2
      
    # Return of closest of two 
    return (b if ((n - a) > (b - n)) else a) 


import random

# shuffle indices within a specified range of a list
def shuffle_slice(a, start, stop):
    i = start
    while (i < (stop-1)):
        idx = random.randrange(i, stop)
        a[i], a[idx] = a[idx], a[i]
        i += 1


def create_batches(gau = False, rep = 0.0):

    # read in sorted list of training set example ids
    with open("train_examples.txt") as myfile1: 
        trainlist = myfile1.readlines()
    trainlist  = [x.rstrip() for x in trainlist]

    # read in list of indices separating examples of different tensor dimensions
    with open("size_indices.txt") as myfile: 
        sindices = myfile.readlines()
    sindices  = [x.rstrip() for x in sindices]

    # shuffle examples within tensor dimension bins
    for i in range(len(sindices) - 1):
        start = int(sindices[i])
        end = int(sindices[i+1])
        shuffle_slice(trainlist, start, end)
        
    # read in list of indices defining effective training batches
    with open("training_indices.txt") as myfile2: 
        indices = myfile2.readlines()
    indices  = [x.rstrip() for x in indices]
    
    # create each training batch
    for i in range(len(indices) - 1):
        start = int(indices[i])
        end = int(indices[i+1])
        xlist = []
        pslist = []
        ylist = []
        for j in range(start, end):

            # load in Patterson map
            new_x = torch.load('patterson_pt_scaled/' + trainlist[j] + '_patterson.pt')
            new_x = torch.unsqueeze(new_x, 0)

            # randomly determine whether to use refined template map or raw previous model's prediction
            replace = False
            r = random.random()
            if r <= rep:
                try:
                    new_x3 = torch.load('predictions_15_angle/' + trainlist[j] + '_refine.pt')
                    replace = True
                except:
                    new_x3 = torch.load('predictions_15_angle/' + trainlist[j] + '.pt')
                    new_x3 = new_x3[0,0,:,:,:]
            else:
                new_x3 = torch.load('predictions_15_angle/' + trainlist[j] + '.pt')
                new_x3 = new_x3[0,0,:,:,:]  

            # if specified, add Gaussian noise to central region of template map
            if gau:
                shape = new_x3.shape
                a = round2(0.85*shape[0])
                b = round2(0.85*shape[1])
                c = round2(0.85*shape[2])
                if replace:
                    gaussian = sqrt(0.001) * torch.randn((a,b,c))
                else:
                    gaussian = sqrt(0.0003) * torch.randn((a,b,c))

                pad_list=[]
                res=shape[2] - c
                pad_list.append(res//2)
                pad_list.append(res -(res//2))
                res=shape[1] - b
                pad_list.append(res//2)
                pad_list.append(res -(res//2))
                res=shape[0] - a
                pad_list.append(res//2)
                pad_list.append(res -(res//2))
                gaussian = torch.nn.functional.pad(gaussian,tuple(pad_list),"constant", 0)

                new_x3.add_(gaussian)
            
            
            new_x3 = torch.unsqueeze(new_x3, 0)
            
            new_xlist = torch.load('res/' + trainlist[j] + '_ps.pt')  

            # concatenate Patterson and template tensors
            new_xcomb = torch.cat((new_x, new_x3), 0)
          
            xlist.append(new_xcomb)
            pslist.append(new_xlist)

            # load in ground truth electron density
            new_y = torch.load('electron_density_pt_scaled/' + trainlist[j] + '_fft.pt')
            new_y = torch.unsqueeze(new_y, 0)
            ylist.append(new_y)

        # combine corresponding tensors
        data_x = torch.stack(xlist)
        data_ps = torch.stack(pslist)
        data_y = torch.stack(ylist)

        # save tensors defining training batch
        torch.save(data_x, 'patterson_pt_scaled/train_' + str(i) + '_patterson.pt')  
        torch.save(data_ps, 'patterson_pt_scaled/train_' + str(i) + '_ps.pt')
        torch.save(data_y, 'electron_density_pt_scaled/train_' + str(i) + '.pt')
        
