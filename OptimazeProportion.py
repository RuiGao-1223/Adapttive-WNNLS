## MSE/wighted MSE
import copy
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import time
import pandas as pd
import scipy
import numpy as np
import warnings
warnings.filterwarnings('ignore')

class decv_MSE(nn.Module):
    def __init__(self, in_fea, out_fea):
        super(decv_MSE, self).__init__()
        self.out = nn.Linear(in_features=in_fea, out_features=out_fea, bias=False)
        # 初始化权重
        self.init_weights()

    def init_weights(self):
        # 初始化权重为细胞类型数量分之一
        initial_weights = np.ones(self.out.in_features) / self.out.in_features
        self.out.weight.data = torch.from_numpy(initial_weights).double().unsqueeze(0)

    def forward(self, x):
        x = self.out(x)
        return x


    
class CustomLoss(nn.Module):
    def __init__(self):
        super(CustomLoss, self).__init__()

    def forward(self, pre_pro, input, target, w=None):
        
        if w is None:
            # Compute squared difference
            mse_loss = torch.sum((input - target)**2)
            limitation = (sum(pre_pro)-1)**2
            loss = mse_loss #+ limitation 
            return loss
            
        else:
            ## weighted MSE
            mse_loss = torch.sum((input - target)**2)

            # Compute weighted squared difference
            mse_vector = ((input - target)**2).view(-1)
            weighted_loss = torch.dot(mse_vector, w)
            limitation = (sum(pre_pro)-1)**2
            loss =  weighted_loss + limitation

            return loss


def deconvolute_pytorch_mse_single_bulk(sc_ref1,bulk1,w, learning_rate,loss_threashold):  

    sc_ref_torch = torch.from_numpy(sc_ref1.to_numpy()).to(dtype=torch.double)  # input x
    bulk_torch = torch.from_numpy(bulk1.to_numpy()).to(dtype=torch.double)
    bulk_torch = bulk_torch / bulk_torch.sum(0)   #Bulk normalization

    infeature = sc_ref1.shape[1]    #3
    outfeature = bulk1.shape[1]    #1

    model = decv_MSE(in_fea=infeature, out_fea=outfeature).double()
    loss_f = CustomLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    consecutive_stable_count = 0


    for step, _ in enumerate(range(10000)):
        pred = model(sc_ref_torch)
        with torch.no_grad():
            pre_pro = list(model.parameters())[0].numpy().ravel().tolist()
        optimizer.zero_grad()
        
        pre_pro = [max(1e-18, x) for x in pre_pro]
        model.out.weight.data = torch.tensor([pre_pro], dtype=torch.double)
        loss= loss_f(pre_pro, pred, bulk_torch, w)
        loss.backward()
        optimizer.step()


        current_loss = loss
        if step > 0 and abs(current_loss-prev_loss)< loss_threashold:
            consecutive_stable_count += 1
        else:
            consecutive_stable_count = 0
        prev_loss = current_loss

        if consecutive_stable_count >= 100:
            # print(f"Optimization stopped at step {step}, loss: {loss}")
            break
   
    return (pre_pro)

def deconvolute_pytorch_mse(sc_ref,bulk,w, learning_rate,loss_threashold):
    pre = []
    for i in range(bulk.shape[1]):
        bulk_single = pd.DataFrame(bulk.iloc[:,i])
        pre_single = deconvolute_pytorch_mse_single_bulk(sc_ref,bulk_single, w, learning_rate,loss_threashold)
        pre.append(pre_single)
    pre_pro= pd.DataFrame(pre,columns=sc_ref.columns, index=bulk.columns)
    return pre_pro