import numpy as np
import torch
import torch.nn as nn
import math
from torch.autograd import Variable
import torch.nn.functional as F
from SNN_layers.spike_neuron import *#
import numpy.random as rd


## readout layer
class readout_integrator_test(nn.Module):
    def __init__(self,input_dim,output_dim,
                 tau_minitializer = 'uniform',low_m = 0,high_m = 4,device='cpu',bias=True,dt = 1):
        """
        Args:
            input_dim(int): input dimension.
            output_dim(int): the number of readout neurons
            tau_minitializer(str): the method of initialization of tau_m
            low_m(float): the low limit of the init values of tau_m
            high_m(float): the upper limit of the init values of tau_m
        """
        super(readout_integrator_test, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.device = device
        self.dt = dt
        self.dense = nn.Linear(input_dim,output_dim,bias=bias)
        self.tau_m = nn.Parameter(torch.Tensor(self.output_dim))
        
        if tau_minitializer == 'uniform':
            nn.init.uniform_(self.tau_m,low_m,high_m)
        elif tau_minitializer == 'constant':
            nn.init.constant_(self.tau_m,low_m)

    
    def set_neuron_state(self,batch_size):
        self.mem = (torch.rand(batch_size,self.output_dim)).to(self.device)
    
    def forward(self,input_spike):
        #synaptic inputs
        d_input = self.dense(input_spike.float())
        # neuron model without spiking
        self.mem = output_Neuron_pra(d_input,self.mem,self.tau_m,self.dt,device=self.device)
        return self.mem
    
class CustomLinear(nn.Module):
    def __init__(self, input_dim, output_dim, nb_non_zero_coeff,use_even_mask=False,even_mask=None):
        super(CustomLinear, self).__init__()

        # Define weight and theta as parameters
        self.W_1 = nn.Parameter(torch.Tensor(output_dim, input_dim))
        self.th_1 = nn.Parameter(torch.Tensor(output_dim))
        self.even_mask = even_mask

        # Initialize parameters using your weight_sampler_strict_number function
        self.W_1, _, self.th_1, _ = self.weight_sampler_strict_number(input_dim, output_dim, nb_non_zero_coeff,use_even_mask)

    def forward(self, x):
        a_1 = F.linear(x, self.W_1.transpose(0, 1))
        return a_1

    def weight_sampler_strict_number(self, input_size, output_size, nb_non_zero, use_even_mask=False):
        # Your implementation of weight_sampler_strict_number
        # Modify self.W_1 and self.th_1 accordingly
        w_0 = rd.randn(input_size,output_size) / np.sqrt(input_size) # initial weight values

        if use_even_mask==False:
            # Gererate the random mask
            # 随机选取nb_non_zero行和nb_non_zero列，将这个区域的mask设为1
            is_con_0 = np.zeros((input_size,output_size),dtype=bool)
            ind_in = rd.choice(np.arange(int(input_size)),size=int(nb_non_zero))
            ind_out = rd.choice(np.arange(int(output_size)),size=int(nb_non_zero))
            is_con_0[ind_in,ind_out] = True
        else:
            assert self.even_mask is not None
            is_con_0 = self.even_mask.transpose(0,1).to(self.th_1.device)
            #create_mask生成的mask形状是(output_dim*branch,input_dim)
            is_con_0.detach().cpu().numpy()

        # Generate random signs
        # 对整个矩阵随机取正负号，无论mask是否为1
        sign_0 = np.sign(rd.randn(input_size,output_size))

        # define the pytorch version
        th = torch.abs(torch.Tensor(w_0) * torch.Tensor(is_con_0))
        #w_sign = torch.Tensor(sign_0,dtype = torch.float32,requires_grad=False)
        w_sign = torch.from_numpy(sign_0).float().requires_grad_(False)

        is_connected = torch.gt(th,0)

        #根据条件 is_connected 在两个不同的值之间选择
        #如果 is_connected 中的元素为 True，则选择 x 参数中的对应元素；否则，选择 y 参数中的对应元素。
        # define the pytorch version
        w = torch.where(is_connected, torch.Tensor(w_sign * th), torch.zeros(input_size,output_size))
        
        w = torch.nn.Parameter(w)
        th = torch.nn.Parameter(th)

        return w,w_sign,th,is_connected
    
    def modify_theta(self,lr,noise,l1):
        theta = self.th_1
        mask_connected = torch.gt(theta, 0).float().to(self.th_1.device)
        noise_update = torch.normal(mean=0,std=noise, size=theta.shape).to(self.th_1.device)
        with torch.no_grad():
            theta.add_(lr * mask_connected * noise_update)
        mask_connected = torch.gt(theta, 0).float().to(self.th_1.device)
        with torch.no_grad():
            theta.add_(- lr * mask_connected * l1)
        return theta

    def rewiring(self,target_nb_connection,epsilon):
        #pytorch version
        with torch.no_grad():
            th = self.th_1
            is_con = torch.gt(th, 0).float()
            n_connected = torch.sum(torch.gt(th, 0).float())
            nb_reconnect = target_nb_connection - n_connected
            nb_reconnect = torch.max(nb_reconnect,torch.tensor(0.0))
            reconnect_candidate_coord = torch.nonzero(torch.eq(is_con, 0))#每一行是一个二维坐标
            n_candidates = reconnect_candidate_coord.shape[0]
            reconnect_sample_id = torch.randperm(n_candidates)[:int(nb_reconnect)]
            reconnect_sample_coord = reconnect_candidate_coord[reconnect_sample_id]
            reconnect_vals = torch.ones(int(nb_reconnect))*epsilon
            th[reconnect_sample_coord[:,0],reconnect_sample_coord[:,1]] = reconnect_vals.to(self.th_1.device)


#DH-SFNN layer
class spike_dense_test_denri_wotanh_R(nn.Module):
    def __init__(self,input_dim,output_dim,tau_minitializer = 'uniform',low_m = 0,high_m = 4,
                 tau_ninitializer = 'uniform',low_n = 0,high_n = 4,vth = 0.5,dt = 1,branch = 2,device='cpu',bias=True,test_sparsity = False,sparsity=0.5,mask_share=1,use_even_mask=False):
        """
        Args:
            input_dim(int): input dimension.
            output_dim(int): the number of readout neurons
            tau_minitializer(str): the method of initialization of tau_m
            low_m(float): the low limit of the init values of tau_m
            high_m(float): the upper limit of the init values of tau_m
            tau_ninitializer(str): the method of initialization of tau_n
            low_n(float): the low limit of the init values of tau_n
            high_n(float): the upper limit of the init values of tau_n
            vth(float): threshold
            branch(int): the number of dendritic branches
            test_sparsity(bool): if testing the sparsity of connection pattern 
            sparsity(float): the sparsity ratio
            mask_share(int): the number of neuron share the same connection pattern 
        """
        super(spike_dense_test_denri_wotanh_R, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.device = device
        self.vth = vth
        self.dt = dt
        if test_sparsity:
            self.sparsity = sparsity 
        else:
            self.sparsity = 1/branch
        
        #group size for hardware implementation
        self.mask_share = mask_share
        self.pad = ((input_dim)//branch*branch+branch-(input_dim)) % branch
        #sparsity
        self.overlap = 1/branch
        self.tau_m = nn.Parameter(torch.Tensor(self.output_dim))
        self.tau_n = nn.Parameter(torch.Tensor(self.output_dim,branch))
        self.test_sparsity = test_sparsity
        
        #the number of dendritic branch
        self.branch = branch
        self.create_mask()
        if use_even_mask:
            self.dense = CustomLinear(input_dim+self.pad,output_dim*branch,(input_dim+self.pad)*output_dim*branch*self.sparsity,use_even_mask=True,even_mask=self.mask)
        else:
            self.dense = CustomLinear(input_dim+self.pad,output_dim*branch,(input_dim+self.pad)*output_dim*branch*self.sparsity)
        
        # timing factor of membrane potential
        if tau_minitializer == 'uniform':
            nn.init.uniform_(self.tau_m,low_m,high_m)
        elif tau_minitializer == 'constant':
            nn.init.constant_(self.tau_m,low_m)
            
            
        # timing factor of dendritic branches
        if tau_ninitializer == 'uniform':
            nn.init.uniform_(self.tau_n,low_n,high_n)
        elif tau_ninitializer == 'constant':
            nn.init.constant_(self.tau_n,low_n)

    

    #init
    def set_neuron_state(self,batch_size):
        #mambrane potential
        self.mem = Variable(torch.rand(batch_size,self.output_dim)).to(self.device)
        self.spike = Variable(torch.rand(batch_size,self.output_dim)).to(self.device)
        # dendritic currents
        if self.branch == 1:
            self.d_input = Variable(torch.rand(batch_size,self.output_dim,self.branch)).to(self.device)
        else:
            self.d_input = Variable(torch.zeros(batch_size,self.output_dim,self.branch)).to(self.device)
        #threshold
        self.v_th = Variable(torch.ones(batch_size,self.output_dim)*self.vth).to(self.device)

    #create connection pattern
    def create_mask(self):
        
        input_size = self.input_dim+self.pad
        self.mask = torch.zeros(self.output_dim*self.branch,input_size).to(self.device)
        for i in range(self.output_dim//self.mask_share):
            seq = torch.randperm(input_size)
            # j as the branch index
            for j in range(self.branch):
                if self.test_sparsity:
                    if j*input_size // self.branch+int(input_size * self.sparsity)>input_size:
                        for k in range(self.mask_share):
                            self.mask[(i*self.mask_share+k)*self.branch+j,seq[j*input_size // self.branch:-1]] = 1
                            self.mask[(i*self.mask_share+k)*self.branch+j,seq[:j*input_size // self.branch+int(input_size * self.sparsity)-input_size]] = 1
                    else: 
                        for k in range(self.mask_share):
                            self.mask[(i*self.mask_share+k)*self.branch+j,seq[j*input_size // self.branch:j*input_size // self.branch+int(input_size * self.sparsity)]] = 1
                else:
                    for k in range(self.mask_share):
                        self.mask[(i*self.mask_share+k)*self.branch+j,seq[j*input_size // self.branch:(j+1)*input_size // self.branch]] = 1
    def apply_mask(self):
        self.dense.weight.data = self.dense.weight.data*self.mask
    def forward(self,input_spike):
        # timing factor of dendritic branches
        beta = torch.sigmoid(self.tau_n)
        padding = torch.zeros(input_spike.size(0),self.pad).to(self.device)
        k_input = torch.cat((input_spike.float(),padding),1)
        #update dendritic currents 
        self.d_input = beta*self.d_input+(1-beta)*self.dense(k_input).reshape(-1,self.output_dim,self.branch)
        #summation of dendritic currents
        l_input = (self.d_input).sum(dim=2,keepdim=False)
        
        #update membrane potential and generate spikes
        self.mem,self.spike = mem_update_pra(l_input,self.mem,self.spike,self.v_th,self.tau_m,self.dt,device=self.device)
        
        return self.mem,self.spike
    
    
#Vanilla SFNN layer
class spike_dense_test_origin(nn.Module):
    def __init__(self,input_dim,output_dim,
                 tau_minitializer = 'uniform',low_m = 0,high_m = 4,vth = 0.5,dt = 4,device='cpu',bias=True):
        """
        Args:
            input_dim(int): input dimension.
            output_dim(int): the number of readout neurons
            tau_minitializer(str): the method of initialization of tau_m
            low_m(float): the low limit of the init values of tau_m
            high_m(float): the upper limit of the init values of tau_m
            vth(float): threshold
        """
        super(spike_dense_test_origin, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.device = device
        self.vth = vth
        self.dt = dt

        self.dense = nn.Linear(input_dim,output_dim)
        self.tau_m = nn.Parameter(torch.Tensor(self.output_dim))

        if tau_minitializer == 'uniform':
            nn.init.uniform_(self.tau_m,low_m,high_m)
        elif tau_minitializer == 'constant':
            nn.init.constant_(self.tau_m,low_m)

    def set_neuron_state(self,batch_size):

        self.mem = Variable(torch.rand(batch_size,self.output_dim)).to(self.device)
        self.spike = Variable(torch.rand(batch_size,self.output_dim)).to(self.device)

        self.v_th = Variable(torch.ones(batch_size,self.output_dim)*self.vth).to(self.device)
    def forward(self,input_spike):
        k_input = input_spike.float()

        d_input = self.dense(k_input)
        self.mem,self.spike = mem_update_pra(d_input,self.mem,self.spike,self.v_th,self.tau_m,self.dt,device=self.device)    
        return self.mem,self.spike
    
#Vanilla SFNN layer without reset 
class spike_dense_test_origin_noreset(nn.Module):
    def __init__(self,input_dim,output_dim,
                 tau_minitializer = 'uniform',low_m = 0,high_m = 4,vth = 0.5,dt = 4,device='cpu',bias=True):
        """
        Args:
            input_dim(int): input dimension.
            output_dim(int): the number of readout neurons
            tau_minitializer(str): the method of initialization of tau_m
            low_m(float): the low limit of the init values of tau_m
            high_m(float): the upper limit of the init values of tau_m
            vth(float): threshold
        """
        super(spike_dense_test_origin_noreset, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.device = device
        self.vth = vth
        self.dt = dt

        self.dense = nn.Linear(input_dim,output_dim)
        self.tau_m = nn.Parameter(torch.Tensor(self.output_dim))

        if tau_minitializer == 'uniform':
            nn.init.uniform_(self.tau_m,low_m,high_m)
        elif tau_minitializer == 'constant':
            nn.init.constant_(self.tau_m,low_m)

    def set_neuron_state(self,batch_size):
        # self.mem = (torch.rand(batch_size,self.output_dim)*self.b_j0).to(self.device)
        self.mem = Variable(torch.rand(batch_size,self.output_dim)).to(self.device)
        self.spike = Variable(torch.rand(batch_size,self.output_dim)).to(self.device)

        self.v_th = Variable(torch.ones(batch_size,self.output_dim)*self.vth).to(self.device)
    def forward(self,input_spike):
        k_input = input_spike.float()
        d_input = self.dense(k_input)
        
        # neural model without reset
        self.mem,self.spike = mem_update_pra_noreset(d_input,self.mem,self.spike,self.v_th,self.tau_m,self.dt,device=self.device)
        
        return self.mem,self.spike

#Vanilla SFNN layer with hard reset 
class spike_dense_test_origin_hardreset(nn.Module):
    def __init__(self,input_dim,output_dim,
                 tau_minitializer = 'uniform',low_m = 0,high_m = 4,vth = 0.5,dt = 4,device='cpu',bias=True):
        """
        Args:
            input_dim(int): input dimension.
            output_dim(int): the number of readout neurons
            tau_minitializer(str): the method of initialization of tau_m
            low_m(float): the low limit of the init values of tau_m
            high_m(float): the upper limit of the init values of tau_m
            vth(float): threshold
        """
        super(spike_dense_test_origin_hardreset, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.device = device
        self.vth = vth
        self.dt = dt

        self.dense = nn.Linear(input_dim,output_dim)
        self.tau_m = nn.Parameter(torch.Tensor(self.output_dim))

        if tau_minitializer == 'uniform':
            nn.init.uniform_(self.tau_m,low_m,high_m)
        elif tau_minitializer == 'constant':
            nn.init.constant_(self.tau_m,low_m)

    def set_neuron_state(self,batch_size):
        # self.mem = (torch.rand(batch_size,self.output_dim)*self.b_j0).to(self.device)
        self.mem = Variable(torch.rand(batch_size,self.output_dim)).to(self.device)
        self.spike = Variable(torch.rand(batch_size,self.output_dim)).to(self.device)

        self.v_th = Variable(torch.ones(batch_size,self.output_dim)*self.vth).to(self.device)
    def forward(self,input_spike):
        k_input = input_spike.float()
        d_input = self.dense(k_input)
        # neural model with hard reset
        self.mem,self.spike = mem_update_pra_hardreset(d_input,self.mem,self.spike,self.v_th,self.tau_m,self.dt,device=self.device)
        
        return self.mem,self.spike


# DH-SFNN for multitimescale_xor task
class spike_dense_test_denri_wotanh_R_xor(nn.Module):
    def __init__(self,input_dim,output_dim,tau_minitializer = 'uniform',low_m = 0,high_m = 4,
                 tau_ninitializer = 'uniform',low_n = 0,high_n = 4,low_n1 = 2,high_n1 = 6,low_n2 = -4,high_n2 = 0,vth = 0.5,dt = 4,branch = 2,device='cpu',bias=True,use_even_mask=False):
        """
        Args:
            input_dim(int): input dimension.
            output_dim(int): the number of readout neurons
            tau_minitializer(str): the method of initialization of tau_m
            low_m(float): the low limit of the init values of tau_m
            high_m(float): the upper limit of the init values of tau_m
            tau_ninitializer(str): the method of initialization of tau_n
            low_n(float): the low limit of the init values of tau_n
            high_n(float): the upper limit of the init values of tau_n
            low_n1(float): the low limit of the init values of tau_n in branch 1 for the beneficial initializaiton
            high_n1(float): the upper limit of the init values of tau_n in branch 1 for the beneficial initializaiton
            low_n2(float): the low limit of the init values of tau_n in branch 2 for the beneficial initializaiton
            high_n2(float): the upper limit of the init values of tau_n in branch 2 for the beneficial initializaiton
            vth(float): threshold
            branch(int): the number of dendritic branches
        """
        super(spike_dense_test_denri_wotanh_R_xor, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        #self.is_adaptive = is_adaptive
        self.device = device
        self.vth = vth
        self.dt = dt
        mask_rate = 1/branch
        self.sparsity = 1/branch

        self.pad = ((input_dim)//branch*branch+branch-(input_dim)) % branch
        #self.dense = nn.Linear(input_dim+self.pad,output_dim*branch,bias=bias)

        self.tau_m = nn.Parameter(torch.Tensor(self.output_dim))
        self.tau_n = nn.Parameter(torch.Tensor(self.output_dim,branch))

        self.branch = branch

        self.create_mask()

        if use_even_mask:
            self.dense = CustomLinear(input_dim+self.pad,output_dim*branch,(input_dim+self.pad)*output_dim*branch*self.sparsity,use_even_mask=True,even_mask=self.mask)
        else:
            self.dense = CustomLinear(input_dim+self.pad,output_dim*branch,(input_dim+self.pad)*output_dim*branch*self.sparsity)

        if tau_minitializer == 'uniform':
            nn.init.uniform_(self.tau_m,low_m,high_m)
        elif tau_minitializer == 'constant':
            nn.init.constant_(self.tau_m,low_m)

        if tau_ninitializer == 'uniform':
            nn.init.uniform_(self.tau_n,low_n,high_n)
        elif tau_ninitializer == 'constant':
            nn.init.constant_(self.tau_n,low_n)
        # init different branch with different scale
        elif tau_ninitializer  == 'seperate':
            nn.init.uniform_(self.tau_n[:,0],low_n1,high_n1)
            nn.init.uniform_(self.tau_n[:,1],low_n2,high_n2)

    

    
    def set_neuron_state(self,batch_size):

        self.mem = Variable(torch.rand(batch_size,self.output_dim)).to(self.device)
        self.spike = Variable(torch.rand(batch_size,self.output_dim)).to(self.device)
        self.d_input = Variable(torch.zeros(batch_size,self.output_dim,self.branch)).to(self.device)

        self.v_th = Variable(torch.ones(batch_size,self.output_dim)*self.vth).to(self.device)

    def create_mask(self):
        input_size = self.input_dim+self.pad
        self.mask = torch.zeros(self.output_dim*self.branch,input_size).to(self.device)
        for i in range(self.output_dim):
            for j in range(self.branch):
                self.mask[i*self.branch+j,j*input_size // self.branch:(j+1)*input_size // self.branch] = 1
    def apply_mask(self):
        self.dense.weight.data = self.dense.weight.data*self.mask
    def forward(self,input_spike):

        beta = torch.sigmoid(self.tau_n)
        padding = torch.zeros(input_spike.size(0),self.pad).to(self.device)
        k_input = torch.cat((input_spike.float(),padding),1)
        self.d_input = beta*self.d_input+(1-beta)*self.dense(k_input).reshape(-1,self.output_dim,self.branch)

        l_input = (self.d_input).sum(dim=2,keepdim=False)
        self.mem,self.spike = mem_update_pra(l_input,self.mem,self.spike,self.v_th,self.tau_m,self.dt,device=self.device)
        
        return self.mem,self.spike