import os
import numpy as np
import torch
import tables
from torch.utils.data import DataLoader, TensorDataset

from scipy.sparse import coo_matrix


import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as dsets
import torch.nn.functional as F
import numpy as np
from torch.autograd import Variable
from torch.optim.lr_scheduler import StepLR,MultiStepLR
import math
#import keras
from torch.utils import data
import matplotlib.pyplot as plt
from datetime import datetime
import os
import argparse
import wandb

parser = argparse.ArgumentParser(description='Training script with command line arguments.')
    
parser.add_argument('--epochs', type=int, default=150, help='Number of training epochs')
parser.add_argument('--branch', type=int, default=2, help='Branch number')
parser.add_argument('--time_steps', type=int, default=100, help='Time steps')
parser.add_argument('--use_even_mask', type=str, help='choose from yes or no')
args = parser.parse_args()

torch.manual_seed(42)

time_steps =100 #delyad timesteps
channel_rate = [0.2,0.6] #spiking rates of high or low
noise_rate = 0.01
channel_size = 20
#lasting time of signal 
coding_time = 10
test_time = 1

perm = torch.randperm(len(channel_rate)**2)
index = perm[:len(channel_rate)**2//2]
label = torch.zeros(len(channel_rate),len(channel_rate))
label[1][0] = 1
label[0][1] = 1

def get_batch():
    """Generate the delayed spiking xor problem dataset"""
    values = torch.rand(batch_size,time_steps,channel_size,requires_grad=False) <= noise_rate
    targets = torch.zeros(time_steps,batch_size,requires_grad=False).int()
    #generate the first signal
    init_pattern = torch.randint(len(channel_rate),size=(batch_size,))
    prob_matrix = torch.ones(coding_time,channel_size,batch_size)*torch.tensor(channel_rate)[init_pattern]
    add_patterns = torch.bernoulli(prob_matrix).permute(2,0,1).bool()

    values[:,:coding_time,:] = values[:,:coding_time,:] | add_patterns
    #generate the position of delayed signal
    position = torch.randint(test_time,size=(batch_size,))
    pattern = torch.randint(len(channel_rate),size=(batch_size,))
    label_t = label[init_pattern,pattern].int()
    prob = torch.tensor(channel_rate)[pattern]
    prob_matrix = torch.ones(coding_time,channel_size,batch_size)*prob 
    add_patterns = torch.bernoulli(prob_matrix).permute(2,0,1).bool()
    #generate the delayed signal
    for i in range(batch_size):
        values[i,time_steps-(position[i]+1)*coding_time:time_steps-(position[i])*coding_time,:] = values[i,time_steps-(position[i]+1)*coding_time:time_steps-(position[i])*coding_time,:] | add_patterns[i]
        targets[time_steps-(position[i]+1)*coding_time:,i] = label_t[i]


    return values, targets.transpose(0,1).contiguous(),position

from SNN_layers.spike_dense import *
from SNN_layers.spike_neuron import *
from SNN_layers.spike_rnn import *
thr_func = ActFun_adp.apply  
is_bias=True
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device:",device)
#vanilla SFNN
class Dense_vanilla(nn.Module):
    def __init__(self, input_size, hidden_dims, output_dim):
        super(Dense_vanilla, self).__init__()

        self.input_size = input_size
        self.dense_1 = spike_dense_test_origin(input_size,hidden_dims,tau_minitializer='uniform',low_m = 0,high_m = 4,
                                    vth= 1,dt = 1,device=device,bias=is_bias)

        self.dense_2 = readout_integrator_test(hidden_dims,output_dim,
                                    dt = 1,device=device,bias=is_bias)


        self.criterion = nn.CrossEntropyLoss()


    def forward(self,input,target,position):
        batch_size,seq_num,input_size= input.shape
        self.dense_1.set_neuron_state(batch_size)
        self.dense_2.set_neuron_state(batch_size)
        d2_output = torch.zeros(batch_size,seq_num,2)
        output = 0
        loss = 0
        total = 0
        correct = 0
        for i in range(seq_num):

            input_x = input[:,i,:]

            #print(input_x.shape)
            mem_layer1,spike_layer1 = self.dense_1.forward(input_x)

            mem_layer2= self.dense_2(spike_layer1)
            d2_output[:,i,:] = mem_layer2.cpu()
            #delayed signal position
            index = i>(time_steps-(position+1)*coding_time)
        
            if (index.sum()>0):
                output = F.softmax(mem_layer2,dim=1)
                loss += self.criterion(output[index],target[index,i].long())
                _, predicted = torch.max(output[index].data, 1)
                labels = target[index,i].cpu()
                predicted = predicted.cpu().t()

                correct += (predicted ==labels).sum()
                total += labels.size()[0]

        return loss,d2_output,correct,total
#DH-SFNN with specified branch number
class Dense_denri(nn.Module):
    def __init__(self, input_size, hidden_dims, output_dim):
        super(Dense_denri, self).__init__()

        self.input_size = input_size

        self.dense_1 = spike_dense_test_denri_wotanh_R(input_size,hidden_dims,tau_minitializer='uniform',low_m = 0,high_m = 4,branch=args.branch,
                                    vth= 1,dt = 1,device=device,bias=is_bias)

        self.dense_2 = readout_integrator_test(hidden_dims,output_dim,
                                    dt = 1,device=device,bias=is_bias)


        self.criterion = nn.CrossEntropyLoss()


    def forward(self,input,target,position):
        batch_size,seq_num,input_size= input.shape
        self.dense_1.set_neuron_state(batch_size)
        self.dense_2.set_neuron_state(batch_size)
        d2_output = torch.zeros(batch_size,seq_num,2)
        output = 0
        loss = 0
        total = 0
        correct = 0
        for i in range(seq_num):

            input_x = input[:,i,:]

            #print(input_x.shape)
            mem_layer1,spike_layer1 = self.dense_1.forward(input_x)

            mem_layer2= self.dense_2(spike_layer1)
            d2_output[:,i,:] = mem_layer2.cpu()

            index = i>(time_steps-(position+1)*coding_time)
            #delayed signal position
            if (index.sum()>0):
                output = F.softmax(mem_layer2,dim=1)
                loss += self.criterion(output[index],target[index,i].long())
                _, predicted = torch.max(output[index].data, 1)
                labels = target[index,i].cpu()
                predicted = predicted.cpu().t()

                correct += (predicted ==labels).sum()
                total += labels.size()[0]

        return loss,d2_output,correct,total


def train(epochs,optimizer,scheduler=None):
    acc_list = []
    best_loss = 150
    log_internel = 100
    path = 'model/'  # .pth'
    #choose
    #name = 'vanilla_sfnn_time'+str(time_steps)
    name = 'denri_sfnn_time'+str(time_steps)
    decay_rate = 0.85
    for epoch in range(epochs):
        train_acc = 0
        sum_sample = 0
        train_loss_sum = 0 
        sum_correct = 0
        model.train()
        if args.use_even_mask=="yes":
            model.dense_1.apply_mask()

        for _ in range(log_internel):
            if args.use_even_mask=="yes":
               model.dense_1.apply_mask()

            data, target,position = get_batch()
            data = data.detach().to(device)
 
            target= target.detach().to(device)
            optimizer.zero_grad()

            loss,output,correct,total = model(data,target,position)

            loss.backward()

            train_loss_sum += loss.item()
            optimizer.step()

            sum_correct += correct.item()
            sum_sample+= total

        if scheduler:
            scheduler.step()

        acc_list.append(train_acc)
        print('lr: ',optimizer.param_groups[0]["lr"])
    
        if train_loss_sum<best_loss*log_internel :
            best_loss = train_loss_sum
            torch.save(model, path+name+str(best_loss/log_internel)[:7]+'-srnn-shd.pth')

        print('log_internel:{:3d}, epoch: {:3d}, Train Loss: {:.4f}, Acc: {:.3f}'.format(log_internel,epoch,train_loss_sum/log_internel,sum_correct/sum_sample), flush=True)
        accuracy = sum_correct/sum_sample
        train_loss = train_loss_sum/log_internel
        wandb.log({'accuracy': accuracy}, step=epoch)
        wandb.log({'train_loss': train_loss}, step=epoch)

    return acc_list
batch_size = 500


hidden_dims = 16 #network size
output_dim = 2

#choose
#model = Dense_vanilla(channel_size,hidden_dims,output_dim)
model = Dense_denri(channel_size,hidden_dims,output_dim)
model.to(device)

learning_rate = 1e-2  

base_params = [                    

                    model.dense_2.dense.weight,
                    model.dense_2.dense.bias,

                    model.dense_1.dense.weight,
                    model.dense_1.dense.bias,

  

                ]



optimizer = torch.optim.Adam([
    {'params': base_params},
    {'params': model.dense_1.tau_m, 'lr': learning_rate},  
    {'params': model.dense_2.tau_m, 'lr': learning_rate}, 
    ],
    lr=learning_rate)

scheduler = StepLR(optimizer, step_size=50, gamma=.1)



epochs =75

wandb.init(project=f"delayed_xor")
wandb.watch(model)
wandb.run.name = f"origin, branch={args.branch}, use_even_mask={args.use_even_mask}"

acc_list = train(epochs,optimizer,scheduler)

