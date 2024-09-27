'''

run: nohup python MyNet_Classification.py --epochs 2 &
                                                                        
'''
import argparse
import os, os.path as osp
import numpy as np
import random
import torch
from torch_geometric.data import DataLoader
#from torch.utils.data import DataLoader
from torch import nn
import sys
sys.path.append('../.')  
sys.path.append('../../.')
from molecular_network.mol_dataset.reduceGraph_dataset import ReduceGraph_Dataset 
from molecular_network.mol_dataset.raw_dataset_4rgnn import Raw_Dataset
from configs.configs import * 
from Nets.NN import NN,ReduceNN
from Nets.AttentiveFP import AFP, ReduceAFP
from Nets.ReduceGNN_pretraining import GNN, RGNN
from Utils.metrics import metrics_C
import time
import matplotlib.pyplot as plt
import numpy as np

torch.manual_seed(123)
start_time = str(time.ctime()).replace(':','-').replace(' ','_')

dirname, filename = os.path.split(os.path.abspath(__file__)) 
line = 'working folder>> ' + dirname + ' >> ' + filename + ' >> '
print(line)

parser = argparse.ArgumentParser()
parser.add_argument('--seed', default=0, help='random seed')
parser.add_argument('--model', default='RGNN', help='')
parser.add_argument('--pretrain', default='', help='if True give checkpoints path')
parser.add_argument('--dataset', default='chembelZINC', help='')
parser.add_argument('--heads', default=2, help='')
parser.add_argument('--epochs', default=1000, help='')
parser.add_argument('--x_latent_dim', default=256, help='')
parser.add_argument('--aggr', default='mean')
parser.add_argument('--dropout', default=0.0)
parser.add_argument('--cuda', default=0)
parser.add_argument('--checkpoints', default=False, action='store_true')
parser.add_argument('--choose_model', default='loss', choices=['loss','AUC'])
args = parser.parse_args()
torch.cuda.set_device(int(args.cuda))
seed = int(args.seed)
model_name = args.model
pretrain = args.pretrain
dataset_name = args.dataset   
epochs = int(args.epochs)
heads = int(args.heads)
x_dim = int(args.x_latent_dim)
aggr = args.aggr
dropout = float(args.dropout)
ckp = args.checkpoints
choose_model = args.choose_model
print('model parameters: ', vars(args))


# load dataset
dataset_option = {'root','raw_name','atom_types','bond_type_num','target_idxs'}
dataset_params = {
    # ... 其他参数 ...
    'root': '/home/data/bvc/PythonWorkSpace/RG-MPNN-main/data/zinc',  # 修改为新的路径
    # ... 其他参数 ...
    'raw_name': 'zinc.csv'
   # 'target_idxs':1,

}

dataset = Raw_Dataset(**dataset_params)
rg_dataset = ReduceGraph_Dataset(**dataset_params)
print('rg_dataset', rg_dataset)
print('dataset', dataset)

# read dataset dim
atom_dim = dataset[0].x.shape[1]
bond_dim = 4
print('bond_dim:', bond_dim)
num_classes =1
print('num_classes:', num_classes)

# read model parameters
batch = 160
print('batch:', batch)
num_passing=3
print('num_passing:', num_passing)
num_passing_before =3
print('num_passing_before:', num_passing_before)
num_passing_pool =2
print('num_passing_pool:', num_passing_pool)
num_passing_after =1
print('num_passing_after:', num_passing_after)
num_passing_mol =2
print('num_passing_mol:', num_passing_mol)

# training parameters
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

if model_name == 'RGNN':
    model = RGNN(in_channels=atom_dim,
                        channels=x_dim,
                        out_channels=num_classes,
                        edge_dim=bond_dim,
                        num_passing_atom=num_passing_before,
                        num_passing_pool=num_passing_pool,
                        num_passing_rg=num_passing_after,
                        num_passing_mol=num_passing_mol,
                        dropout= dropout
                        ).to(device)

base_lr =0.0001
print('base_lr:', base_lr)
factor =0.95
print('factor:', factor)
patience = 10
print('patience:', patience)
pos_weight = 1.2
print('pos_weight:', pos_weight)
optimizer = torch.optim.Adam(model.parameters(), lr=base_lr)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',
                                                        factor=factor, patience=patience,
                                                        min_lr=0.00001)  # min_lr=0.00001

import torch
import torch.nn.functional as F
batch_size = 160
print('batch_size:', batch_size)
import torch.nn as nn

triplet_loss = nn.TripletMarginLoss(margin=0.0, p=2)

import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import MessagePassing
from misc.loss import HardNegativeContrastiveLoss, ContrastiveLoss


# model: RGNN 模型
# optimizer: 用于优化模型的优化器
# device: 用于指定设备（例如 'cuda' 或 'cpu')
# dataloader 和 rg_dataloader: 数据加载器
def train(dataloader, rg_dataloader,model, optimizer, device):
    # --------* train *---------
    model.train()
    state_dict = model.state_dict()
    loss_all = 0

    for data,rg_data in zip(dataloader,rg_dataloader):
        data = data.to(device)
        rg_data = rg_data.to(device)
        optimizer.zero_grad()
        out = model(data, rg_data)[0]  # 获取模型的原始输出
        rg_out = model(data, rg_data)[1]
        loss_fun = HardNegativeContrastiveLoss(margin=0.2)
        loss = loss_fun(out, rg_out)
        loss.backward()
        loss_all += loss.item()
        optimizer.step()
    return state_dict, loss_all / len(dataloader.dataset)
# 调用训练函数
def test(dataloader, rg_dataloader, model, optimizer, device):
    model.eval()
    state_dict = model.state_dict()
    loss_all = 0

    for data, rg_data in zip(dataloader, rg_dataloader):
        data = data.to(device)
        rg_data = rg_data.to(device)
        optimizer.zero_grad()
        out = model(data, rg_data)[0]  # 获取模型的原始输出
        rg_out = model(data, rg_data)[1]
        loss_fun = HardNegativeContrastiveLoss(margin=0.2)
        loss = loss_fun(out, rg_out)
        loss.backward()
        loss_all += loss.item()
        optimizer.step()
    return state_dict, loss_all / len(dataloader.dataset)



# save checkpoint files
if ckp:
    ckp_folder = osp.join(dirname, 'checkpoints', dataset_name)
    os.makedirs(ckp_folder) if not os.path.exists(ckp_folder) else None


# random shuffle two datasets
random.seed(seed)
index = [i for i in range(len(dataset))]
random.shuffle(index)
dataset = dataset[index]
print('len(dataset):', len(dataset))
rg_dataset = rg_dataset[index]
print('len(rg_dataset):', len(rg_dataset))
print('top five items', index[:5])

split = len(dataset) // 10
print('len(dataset):', len(dataset))
train_dataset = dataset[:-2*split]
print('len(train_dataset):', len(train_dataset))
print('train_dataset:', train_dataset)

val_dataloader = DataLoader(dataset[-2*split:-split], batch_size=batch) # 10%
print('val_dataloader:', val_dataloader)
test_dataloader = DataLoader(dataset[-split:], batch_size=batch) # 10%
train_rg_dataset = rg_dataset[:-2*split]

val_rg_dataloader = DataLoader(rg_dataset[-2*split:-split], batch_size=batch) # 10%
test_rg_dataloader = DataLoader(rg_dataset[-split:], batch_size=batch) # 10%

# 存储训练和验证损失
train_losses = []
val_losses = []
best_val_loss = float('inf')
patience = 10  # 早停的耐心值
counter = 0

for epoch in range(epochs):
    lr = scheduler.optimizer.param_groups[0]['lr']

    random.seed(epoch)
    index = [i for i in range(len(train_dataset))]
    random.shuffle(index)
    train_dataset = train_dataset[index]
    train_rg_dataset = train_rg_dataset[index]
    train_dataloader = DataLoader(train_dataset, batch_size=batch) # 80%
    train_rg_dataloader = DataLoader(train_rg_dataset, batch_size=batch) # 80%
    state_dict, loss = train(epoch, train_dataloader, train_rg_dataloader, model, optimizer, device)
    train_losses.append('%.2f' % loss)
    print(f'Epoch {epoch + 1}/{epochs}, Train Loss: {loss:.4f}')

    state_dict_test, val_loss = test(epoch, val_dataloader, val_rg_dataloader, model, optimizer, device)
    val_losses.append('%.2f' % val_loss)
    print(f'Epoch {epoch + 1}/{epochs}, Validation Loss: {val_loss:.4f}')
    # 保存最佳模型
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        counter = 0  # 重置计数器
        best_model_path = osp.join(ckp_folder, model_name + '_best_model.ckp')
        torch.save(state_dict, best_model_path)
        print(f'New best model saved at epoch {epoch + 1}')
    else:
        counter += 1
        print(f'No improvement, counter: {counter}')
    # 应用早停策略
    if counter >= patience:
        print(f'Early stopping at epoch {epoch + 1}')
        break

    # 保存检查点
    ckp_path = osp.join(ckp_folder, model_name + f'_epoch_{epoch + 1}_{start_time}.ckp')
    torch.save(state_dict, ckp_path)


