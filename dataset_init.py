from torch.utils.data import Dataset, DataLoader

import torch 
import torch.nn as nn
import torch.fft
import torch.cuda
import numpy as np

import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

import time


import random
def shuffle_slice(a, start, stop):
    i = start
    while (i < (stop-1)):
        idx = random.randrange(i, stop)
        a[i], a[idx] = a[idx], a[i]
        i += 1


def create_batches():
    
    with open("train_examples.txt") as myfile1: 
        trainlist = myfile1.readlines()
    trainlist  = [x.rstrip() for x in trainlist]
    
    with open("size_indices.txt") as myfile: 
        sindices = myfile.readlines()
    sindices  = [x.rstrip() for x in sindices]
    
    for i in range(len(sindices) - 1):
        start = int(sindices[i])
        end = int(sindices[i+1])
        shuffle_slice(trainlist, start, end)
        

    with open("training_indices.txt") as myfile2: 
        indices = myfile2.readlines()
    indices  = [x.rstrip() for x in indices]
    

    for i in range(len(indices) - 1):
        start = int(indices[i])
        end = int(indices[i+1])
        xlist = []
        pslist = []
        ylist = []
        
        for j in range(start, end):
            new_x = torch.load('patterson_pt_scaled/' + trainlist[j] + '_patterson.pt')
            new_x = torch.unsqueeze(new_x, 0)
            
            new_xlist = torch.load('res/' + trainlist[j] + '_ps.pt')  
            
            
            xlist.append(new_x)
            pslist.append(new_xlist)
            new_y = torch.load('electron_density_pt_scaled/' + trainlist[j] + '_fft.pt')
            new_y = torch.unsqueeze(new_y, 0)
            ylist.append(new_y)
        
        data_x = torch.stack(xlist)
        data_ps = torch.stack(pslist)
        data_y = torch.stack(ylist)
        torch.save(data_x, 'patterson_pt_scaled/train_' + str(i) + '_patterson.pt')  
        torch.save(data_ps, 'patterson_pt_scaled/train_' + str(i) + '_ps.pt')
        torch.save(data_y, 'electron_density_pt_scaled/train_' + str(i) + '.pt')
        