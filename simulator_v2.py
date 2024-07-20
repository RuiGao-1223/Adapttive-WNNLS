import warnings
warnings.filterwarnings('ignore')

from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
import random
import os
import scanpy as sc

import copy
import time
from collections import Counter

""" two parts:
 1. simulator qiuyu 写的
 2. simulator for deconvolute
"""


# PART 1
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun  7 23:04:42 2021

@author: qy
"""

import numpy as np
import random
import pandas as pd
from scipy.stats import bernoulli

def dropletFormation(sample_component, cell_capture_rate=0.5, tot_droplets=80000):
    '''
    Given cell type composition in the loaded sample, return the composition of output droplets. 

    Parameters
    ----------
    sample_component : list of list
        size of cell types in HTO samples. each element records the sizes of each cell type in a HTO. elements are equal in length.
    cell_capture_rate : float, optional
        cell capture rate in droplets
    tot_droplets : int, optional
        num of droplets in one-run of 10x. The default is 80000.
    
    Returns
    -------
    droplet_qualified : pandas data frame 
        output droplets with cells and beads simultaneously

    '''
    
    num_HTO = len(sample_component)
    num_celltype = len(sample_component[0])
    
    num_cells_eachHTO = [sum(x) for x in sample_component]
    tot_cells = sum(num_cells_eachHTO)
    
    hto = [[i]*sum(sample_component[i]) for i in range(num_HTO)]
    HTO_tags = [x for sub in hto for x in sub]
    
    ct_list = [[[i]*sample_component[hto][i] for i in range(num_celltype)] for hto in range(num_HTO)]
    celltypes = [x for hto in ct_list for sub in hto for x in sub]
    
    sampleDroplet = random.choices(range(tot_droplets),k=tot_cells)
    val, cnt = np.unique(sampleDroplet, return_counts = True)
    
    droplet2cnt = pd.Series(cnt, index=val)
    
    cell_info = pd.DataFrame({'cnt':[droplet2cnt[sampleDroplet[i]] for i in range(tot_cells)],
                             'droplet':sampleDroplet,
                             'celltype':celltypes,
                             'HTO':HTO_tags},index=range(tot_cells)) 
    
    droplets = []
    multiplet_cell_pool = []
    for i in range(tot_cells):
        if i in multiplet_cell_pool:
            continue
            
        if cell_info.loc[i,'cnt'] ==1:
            droplets.append([i])
        else:
            cell_together = cell_info.index[cell_info['droplet']==cell_info.loc[i,'droplet']].values.tolist()
            multiplet_cell_pool += cell_together
            droplets.append(cell_together)
    
    
    droplet_info = pd.DataFrame({'component':droplets,
                                 'num_cells':[len(x) for x in droplets],
                                 'HTO':[list(cell_info.loc[x,'HTO']) for x in droplets],
                                 'cell_type':[list(cell_info.loc[x,'celltype']) for x in droplets],
                                 'beads':bernoulli.rvs(cell_capture_rate, size=len(droplets))})
    
    droplet_qualified = droplet_info.loc[droplet_info['beads']==1,['component','num_cells','HTO','cell_type']]
    #droplet_qualified.reset_index(drop=True,inplace=True)
    
    # print('Please check if the generated true singlet/dobulet nubmers meet the need:')
    # print('--num of valid droplets:', droplet_qualified.shape[0])
    # print('--num of true singlets:',sum(droplet_qualified['num_cells'] == 1))
    # print('--num of true doublets:',sum(droplet_qualified['num_cells'] == 2))
    return droplet_qualified

import copy
import os
import scanpy as sc
from matplotlib import pyplot as plt




def absoluteLS(sample_component, logLS_m, logLS_std):
    '''
    Given the absolute library-size distribution, 
    generate the absolute number of mRNA molecules in a cell.

    Parameters
    ----------
    sample_component : list of list
        size of cell types in HTO samples. each element records the sizes of each cell type in a HTO. elements are equal in length.
    logLS_m : list
        mean of logLS in each cell types.
    logLS_std : list
        std of logLS in each cell types.

    Returns
    -------
    array
        mRNA amounts.

    '''
    logLS_list = []
    for hto in sample_component:
        for cidx, num_cells in enumerate(hto):
            logLS = np.random.normal(logLS_m[cidx], logLS_std[cidx], num_cells)
            logLS_list.extend(list(logLS))
        
    return (10**np.array(logLS_list)).astype(int)


def sampling(x, base = 10000, base_cr = 0.1, decay_coef = 0.85):
    
    decay = decay_coef**np.log2(x/base)

    return (x*base_cr*decay).astype(int)


def cell_probV(alpha_arr, sample_component):
    '''
    sampling probabilistic vectors from Dirichlet Distribution.

    Parameters
    ----------
    alpha_arr : array
        Parameter of dirichlet for each cell type.
    sample_component : list of list
        size of cell types in HTO samples. each element records the sizes of each cell type in a HTO. elements are equal in length.

    Returns
    -------
    None.

    '''
    
    pvec_list = []
    for hto in sample_component:
        for cidx, num_cells in enumerate(hto):
            pvec_list.append(np.random.dirichlet(alpha_arr[cidx], size = num_cells))
    
    return np.concatenate(pvec_list)
   

def sampling_mRNA(abs_vec, n):
    
    #t0 = time.time()
    ngene = len(abs_vec)
    abs_vec_list = [[i]*abs_vec[i] for i in range(ngene)]
    abs_vec_1hot = [val for sub in abs_vec_list for val in sub]
    #print(time.time()-t0)
    sam_vec_1hot = np.random.choice(abs_vec_1hot, n, replace = False)
    #print(time.time()-t0)
    
    val,cnt = np.unique(sam_vec_1hot, return_counts=True)
    
    val_series = pd.Series([0]*ngene,index=range(ngene))
    val_series[val] = cnt
    
    #sam_vec = [sum(sam_vec_1hot==i) for i in range(ngene)]
    sam_vec = val_series.values.tolist()
    #print(time.time()-t0)
    
    return sam_vec


import time
import multiprocessing as mp
from multiprocessing import shared_memory


def sub_simulate_UMIs(shm_name, pmat_shape, droplets):
    
    existing_shm = shared_memory.SharedMemory(name=shm_name)
    pmat = np.ndarray(pmat_shape, dtype=np.float64, buffer=existing_shm.buf)
    
    UMI_counts = []
    t = 0
    for d in droplets.index:# for i in range(start,end):
        #if t % 100 == 0:
        #    print(t)
        t += 1
        #d = droplets.index[i]
        #t0 = time.time()
        totm = droplets['cell_mRNA'][d]
        comp = droplets['component'][d]
        num = droplets['num_cells'][d]
        
        abs_counts = np.array([np.random.multinomial(totm[i], pmat[comp[i]]) for i in range(num)]).sum(0)
        
        ls = droplets['ls'][d]
        UMI_counts.append(sampling_mRNA(abs_counts, ls))
        #print('one run: ', time.time()-t0)
    
    existing_shm.close()
    
    return UMI_counts


def simulate_Droplets(cell_pmat, droplets):
    '''
    generate observed UMI matrix.

    Parameters
    ----------
    cell_pmat : TYPE
        DESCRIPTION.
    logLS_m : list
        DESCRIPTION.

    Returns
    -------
    UMI_mat : TYPE
        DESCRIPTION.

    '''
        
    # droplet_w = pd.Series([[1] if droplets['num_cells'][d] == 1 else get_w(droplets['cell_type'][d],logLS_m) for d in droplets.index],
    #                      index = droplets.index)
    #
    # droplet_plist = [np.matmul(np.array(droplet_w[d]).reshape(1,-1), cell_pmat[ droplets['component'][d] ])[0,:] for d in droplets.index]
    #
    # UMI_mat = np.array([np.random.multinomial(droplets['ls'].values[i], droplet_plist[i]) for i in range(droplets.shape[0])])
    
    idx = [1000*i for i in range(int(len(droplets)/1000))]
    idx.append(len(droplets))
    
    pmat_shape = cell_pmat.shape
    
    # creat a sahred memory block
    shm = shared_memory.SharedMemory(create=True, size=cell_pmat.nbytes)
    # # Now create a NumPy array backed by shared memory
    np_array = np.ndarray(pmat_shape, dtype=np.float64, buffer=shm.buf)
    np_array[:] = cell_pmat[:]  # Copy the original data into shared memory
    # del cell_pmat
    
    t0 = time.time()
    pool = mp.Pool()  
    compact_re = [pool.apply_async(sub_simulate_UMIs, (shm.name, pmat_shape, droplets[idx[i] : idx[i+1]])) for i in range(len(idx)-1)]
    pool.close()
    pool.join()
    # print('multiprocessing: ', time.time()-t0)

    UMI_list = [item.get() for item in compact_re]
    UMI_flat_list = [item for sub in UMI_list for item in sub]

    return np.array(UMI_flat_list)



## part2 
def generate_mat(sample_component,cell_capture_rate, celltype_names, logLS_m, logLS_std, alpha_arr, gene_name):
    droplets = dropletFormation(sample_component,cell_capture_rate)
    UMI_info = copy.deepcopy(droplets)
    cell_LS = absoluteLS(sample_component, logLS_m, logLS_std)                  # tot.mRNA per cell 
    UMI_info['cell_mRNA'] = [list(cell_LS[d]) for d in UMI_info['component']]   # tot.mRNA per cell in a droplet
    UMI_info['tot_mRNA'] = [sum(term) for term in UMI_info['cell_mRNA']]        # tot.mRNA per droplet
    UMI_info['ls'] = [sampling(N) for N in UMI_info['tot_mRNA']]                # tot.UMI per drop;let
    UMI_info['trueR']  = [sum(np.array(UMI_info.loc[d,'cell_type'])*np.array(UMI_info.loc[d,'cell_mRNA']))/sum((1-np.array(UMI_info.loc[d,'cell_type']))*np.array(UMI_info.loc[d,'cell_mRNA'])) for d in
    UMI_info.index]

    cell_pmat = cell_probV(alpha_arr, sample_component)   # p_vec per droplet
    t0 = time.time()
    UMI_mat = simulate_Droplets(cell_pmat, UMI_info)
    # print(time.time() - t0)

    matrix = pd.DataFrame(np.array(UMI_mat)).T
    matrix = matrix.set_index(gene_name).T
    matrix['cell_num'] = list(UMI_info['num_cells'])
    matrix['cell_type'] = list(UMI_info['cell_type'])
    matrix['HTO']=UMI_info['HTO'].values
    matrix=matrix[matrix['cell_num']==1]
    # matrix=matrix.drop['cell_type']
    matrix['cell_type'] = matrix['cell_type'].apply(lambda x: ', '.join(map(str, x)))
    matrix['HTO'] =  matrix['HTO'].apply(lambda lst: [x + 1 for x in lst])
    matrix['HTO'] = matrix['HTO'].apply(lambda x: "sample"+', '.join(map(str, x)))
    coded = sorted(list(set(matrix['cell_type'])))
    # celltype_names= ['mNK','Bcell','CD4T']
    ct_map_dict = dict(zip(coded, celltype_names[0:len(coded)]))
    matrix['cell_type'] = matrix['cell_type'].map(ct_map_dict)

    return matrix


def generate_sc(sample_component,cell_capture_rate, celltype_names, logLS_m, logLS_std, alpha_arr, gene_name):
    matrix = generate_mat(sample_component,cell_capture_rate, celltype_names, logLS_m, logLS_std, alpha_arr, gene_name)
    sc_m = matrix.iloc[:,0:len(gene_name)]
    sc_ct=matrix.iloc[:,len(gene_name)]
    sc_m = matrix.iloc[:,0:len(gene_name)]
    sc_info = matrix[['cell_type']]
    sc_info['sampleID'] = matrix['HTO']
    sc_m = sc_m.reset_index(drop = True)
    sc_ct = sc_ct.reset_index(drop = True)
    return sc_m, sc_info

def generate_bulk(sample_component,cell_capture_rate, celltype_names, logLS_m, logLS_std, alpha_arr, gene_name):
    matrix = generate_mat(sample_component,cell_capture_rate, celltype_names, logLS_m, logLS_std, alpha_arr, gene_name)
    samples = set(matrix['HTO'])
    bulk_gt_list = []
    bulk = []
    sample_label = []
    for i in samples:
        bulk_single = matrix[matrix['HTO'] == i]
        # print(Counter(bulk_single['cell_type']))
        bulk_gt_list.append(list(Counter(bulk_single['cell_type']).values()))
        sample_label.append(i)
        bulk.append(bulk_single.iloc[:,0:len(gene_name)].sum())
    bulk = pd.DataFrame(bulk,index = sample_label)
    gt = pd.DataFrame(bulk_gt_list,index = sample_label,columns = celltype_names)
    gt = gt.div(gt.sum(axis=1), axis=0)
    return bulk, gt