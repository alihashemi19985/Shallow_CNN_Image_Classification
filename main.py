import pandas as pd 
import numpy as np
import torch
import matplotlib.pyplot as plt
import torch.nn.functional as F
import torch.nn.init as init
import torch.nn as nn
from tqdm import tqdm
from torchvision import utils,datasets
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import random_split, Dataset, DataLoader
from torch.optim.lr_scheduler import ExponentialLR, CosineAnnealingWarmRestarts
from torchsummary import summary
import logging
import os 

def fit(clf,
        train_loader,
        optimizer,
        criterian,
        scheduler):

    clf.train()
    training_loss_running = 0
    training_correct_running = 0
    total = 0
    counter = 0
    for i,data in enumerate(train_loader):
        counter += 1
        data,label = data[0],data[1]
        #data = data.reshape(-1,28*28)
        total += label.size(0)
        optimizer.zero_grad()
        out = clf(data) 
        loss = criterian(out,label)
        training_loss_running += loss.item()
        _,pred = torch.max(out.data,1)
        training_correct_running += (pred == label).sum().item()
        loss.backward()
        optimizer.step()
    scheduler.step() 
    train_loss = training_loss_running / counter
    train_accuracy = 100. * training_correct_running / total
    return train_loss, train_accuracy     


def validation (clf,validation_loader,criterian,epoch):
    clf.eval()
    valid_loss_running = 0
    valid_acc_running = 0
    total = 0
    counter = 0
    for i,data in enumerate(validation_loader):
        counter += 1
        data,label = data[0],data[1]
        #data = data.reshape(-1,28*28)
        total += label.size(0)
        out = clf(data)
        loss = criterian(out,label)
        valid_loss_running += loss.item()
        _,pred = torch.max(out.data,1)
        valid_acc_running += (pred == label).sum().item()

    valid_loss = valid_loss_running / counter
    valid_acc = 100. * valid_acc_running / total  
    chk.save(valid_acc,'chk',epoch,clf)
      

    return valid_loss,valid_acc


        
def train (hyparam,train_loader,val_loader):

    clf = SCNNB()
    #optimizer = torch.optim.Adam(clf.parameters(), lr =hyparam['lr'])
    optimizer = torch.optim.SGD(clf.parameters(), lr=hyparam['lr'],momentum=0.9, weight_decay=0.001)
    criterian = nn.CrossEntropyLoss()
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=50, T_mult=1)

    train_loss =[]
    train_acc =[]
    val_loss =[]
    val_acc = []
    
    for epoch in range(hyparam['epoch']):
        print(f"Epoch {epoch+1} of {hyparam['epoch']}")
        training_loss,training_acc = fit(clf,train_loader,optimizer,criterian,scheduler)
        validation_loss,validation_acc = validation(clf,val_loader,criterian,epoch)

        train_loss.append(training_loss)
        train_acc.append(training_acc)

        val_loss.append(validation_loss)
        val_acc.append(validation_acc)

        

        logger.info(f" Epoch: {epoch + 1}, Train Loss: {training_loss:.4f}, Train Acc: {training_acc:.2f},\
         Val Loss: {validation_loss:.4f}, Val Acc: {validation_acc:.2f}")
       
        print(f"Train Loss: {training_loss:.4f}, Train Acc: {training_acc:.2f},\
         Val Loss: {validation_loss:.4f}, Val Acc: {validation_acc:.2f}")



      
    return clf,train_loss,train_acc,val_loss,val_acc
        

hparams = {'batch_size': 256, 'lr': 0.6e-1, 'epoch': 50} #6e-4
clf,train_loss,train_acc,val_loss,val_acc = train(hparams,train_loader,val_loader) 