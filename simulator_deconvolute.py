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
    
    print('Please check if the generated true singlet/dobulet nubmers meet the need:')
    print('--num of valid droplets:', droplet_qualified.shape[0])
    print('--num of true singlets:',sum(droplet_qualified['num_cells'] == 1))
    print('--num of true doublets:',sum(droplet_qualified['num_cells'] == 2))
    return droplet_qualified

import copy
import os
import scanpy as sc
from matplotlib import pyplot as plt

def generateUMIobs(sample_component, droplets, para, output, display=True):

    #LS_base = para['LS_base']
    R_setting = para['R_setting']
    alpha_arr = para['Alpha']
    logLS_std_list = para['logLS_std']
    logLS_m_list = para['logLS_m']
    
    if len(alpha_arr) > 2:
        display=False


    for ridx in range(len(R_setting)):
        
        R_set = '_'.join([str(r) for r in R_setting[ridx]]) # 
        
        savepath = os.path.join(output,R_set)
        if not os.path.exists(savepath):
            os.mkdir(savepath)
        
        print('total mRNA ratio R =',str(R_set))
        logLS_m = logLS_m_list[ridx]
        logLS_std = logLS_std_list[ridx]
        #logLS_m = [np.log10(LS_base), np.log10(R_set*LS_base)]

        '''
        Simulate mRNA sampling process within droplets and generate UMI matrix.
        '''
        UMI_info = copy.deepcopy(droplets)
        cell_LS = absoluteLS(sample_component, logLS_m, logLS_std)
        UMI_info['cell_mRNA'] = [list(cell_LS[d]) for d in UMI_info['component']]   
        UMI_info['tot_mRNA'] = [sum(term) for term in UMI_info['cell_mRNA']]
        #UMI_info['ls'] = [LS_recession(N) for N in UMI_info['tot_mRNA'] ]
        UMI_info['ls'] = [sampling(N) for N in UMI_info['tot_mRNA'] ]

        UMI_info['trueR']  = [sum(np.array(UMI_info.loc[d,'cell_type'])*np.array(UMI_info.loc[d,'cell_mRNA']))/sum((1-np.array(UMI_info.loc[d,'cell_type']))*np.array(UMI_info.loc[d,'cell_mRNA'])) for d in UMI_info.index]
        UMI_info.to_csv(os.path.join(savepath,'UMI_info.csv'),header=True,index=None)

        print('sampling probabilistic templtes...')
        cell_pmat = cell_probV(alpha_arr, sample_component)

        t0 = time.time()
        UMI_mat = simulate_Droplets(cell_pmat, UMI_info)
        print('sampling time:',time.time() - t0)

        '''
        Extract UMI_info of interest.
        '''
        adata = sc.AnnData(UMI_mat)

        droplet_type = UMI_info['guasi-GMMdemux'].unique()
        for i,d in enumerate(droplet_type):
            adata[UMI_info['guasi-GMMdemux']==d,:].write_h5ad(os.path.join(savepath,droplet_type[i]+'.h5ad')) 

        print('data saved. done!')


        if display:

            cts = [term.split('_')[0] for term in droplet_type if len(term.split('_'))==1]
            cts.remove('Multiplet')
            print('Celltypes:', cts)

            '''
            Display properties of simulated data.
            '''
            fig, axs =plt.subplots(2,2,figsize=(18,8),dpi=512)

            lb = np.min(np.log10(UMI_info['ls']))-0.05
            ub = np.max(np.log10(UMI_info['ls']))+0.05

            plt.subplot(2,3,1)
            plt.hist( np.log10(UMI_info['ls']),30)
            plt.title('all UMI_info',fontsize=16)
            plt.ylabel('Frequency',fontsize=14)

            plt.xlim(lb,ub)

            plt.subplot(2,3,4)
            for d in droplet_type:
                plt.hist(np.log10(UMI_info['ls'][UMI_info['guasi-GMMdemux']==d]),
                         30,alpha=0.6,label=d)
            plt.title('GMMdemux',fontsize=16)
            plt.legend()
            plt.xlabel('log10LS',fontsize=14)
            plt.ylabel('Frequency',fontsize=14)
            plt.xlim(lb,ub)

            plt.subplot(2,3,2)
            n1 = np.log10(UMI_info['ls'][UMI_info['guasi-GMMdemux']==cts[0]])
            n2 = np.log10(UMI_info['ls'][UMI_info['guasi-GMMdemux']=='2mix_'+cts[0]+'_'+cts[0]])
            (cnt, _, _) = plt.hist(n1,30,alpha=0.6,label=cts[0])
            plt.hist(n2,30,alpha=0.6,label='2mix_'+cts[0])
            plt.plot([np.mean(n1),np.mean(n1)],[0,np.max(cnt)*1.2],label=cts[0]+' Mean')
            plt.plot([np.mean(n2),np.mean(n2)],[0,np.max(cnt)*1.2],label='2mix Mean')
            plt.plot([np.mean(n1)+np.log10(2),np.mean(n1)+np.log10(2)],[0,np.max(cnt)*1.2],label='2x'+cts[0]+' Mean')
            plt.title('GMMdemux-'+cts[0],fontsize=16)
            plt.legend()
            plt.xlim(lb,ub)

            plt.subplot(2,3,5)
            a1 = np.log10(UMI_info['ls'][UMI_info['guasi-GMMdemux']==cts[1]])
            a2 = np.log10(UMI_info['ls'][UMI_info['guasi-GMMdemux']=='2mix_'+cts[1]+'_'+cts[1]])
            (cnt, _, _) = plt.hist(a1,30,alpha=0.6,label=cts[1])
            plt.hist(a2,30,alpha=0.6,label='2mix_'+cts[1])
            plt.plot([np.mean(a1),np.mean(a1)],[0,np.max(cnt)*1.2],label=cts[1]+' Mean')
            plt.plot([np.mean(a2),np.mean(a2)],[0,np.max(cnt)*1.2],label='2mix Mean')
            plt.plot([np.mean(a1)+np.log10(2),np.mean(a1)+np.log10(2)],[0,np.max(cnt)*1.2],label='2x'+cts[1]+' Mean')
            plt.title('GMMdemux-'+cts[1],fontsize=16)
            plt.legend()
            plt.xlabel('log10LS',fontsize=14)
            plt.xlim(lb,ub)

            dbl=[term for term in droplet_type if len(np.unique(term.split('_')))==3][0]

            plt.subplot(2,3,3)
            EstiDoublet_R = UMI_info['trueR'][UMI_info['guasi-GMMdemux']==dbl]
            EstiDoublet_R_m = np.mean(np.log2(EstiDoublet_R))
            (cnt, _, _) = plt.hist(np.log2(EstiDoublet_R),30)
            plt.plot([EstiDoublet_R_m,EstiDoublet_R_m],[0,np.max(cnt)*1.01],label='mean')
            plt.plot([np.log2(R_set), np.log2(R_set)],[0,np.max(cnt)*1.2],label='truth')
            plt.title('GMMdemux-doublets',fontsize=16)
            plt.legend()

            plt.subplot(2,3,6)
            trueDoublet_idx = [d for d in UMI_info.index if UMI_info.loc[d,'guasi-GMMdemux']==dbl and UMI_info.loc[d,'num_cells'] == 2]
            TrueDobulet_R = UMI_info['trueR'][trueDoublet_idx]
            TrueDobulet_R_m = np.mean(np.log2(TrueDobulet_R))
            (cnt, _, _) = plt.hist(np.log2(TrueDobulet_R),30)
            plt.plot([TrueDobulet_R_m,TrueDobulet_R_m],[0,np.max(cnt)*1.01],label='mean')
            plt.plot([np.log2(R_set),np.log2(R_set)],[0,np.max(cnt)*1.2],label='truth')
            plt.title('True-doublets',fontsize=16)
            plt.legend()
            plt.xlabel('log2R',fontsize=14)
            fig.subplots_adjust(top=0.85)
            plt.tight_layout()
            plt.savefig(os.path.join(savepath,'UMIdist.png'))
            plt.show()


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

def LS_recession(tot_mRNA, logit_L=100000, logit_k=1e-5):
    '''
    Given N, the absolute number of mRNA molecules in a droplet, 
    return the number of sampled mRNA molecules in the droplet.

    Parameters
    ----------
    tot_mRNA : int
        how many mRNA molecules there are in a droplet.
    logit_L : int, optional
        the curve's maximum value (the saturation value of mRNA-capturing beads). The default is 1e5.
    logit_k : float, optional
        the logistic growth rate or steepness of the curve. The default is 1e-5.
        
    Returns
    -------
    array
        num of sampled mRNA in the droplet.

    '''
    # recession function
    #n_sampling_UMI = 2*logit_L/(1+np.exp(-logit_k*(tot_mRNA))) - logit_L
    n_sampling_UMI = logit_L *(1-np.exp(-0.5*logit_k*tot_mRNA))
    
    return n_sampling_UMI.astype(int)


def sampling(x, base = 10000, base_cr = 0.1, decay_coef = 0.85):
    
    decay = decay_coef**np.log2(x/base)

    return (x*base_cr*decay).astype(int)


def estimate_total_mRNA(tot_UMI, logit_L = 100000, logit_k = 1e-5):
    
    tot_mRNA = -1/logit_k*np.log(2*logit_L/(tot_UMI+logit_L)-1)
    
    return tot_mRNA


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




#from scipy.stats import beta
# super-parameters: Alpha, size, absoluteLS-m/std for each cell type
# truth: R = absoluteLS-m1 / absoluteLS-m2

def generate_alpha(precision, num_genes, p_center, n_diff_genes, dispersion):
    
    p_vec = pd.Series(np.abs(np.random.normal(p_center, dispersion*p_center, num_genes)), index = range(num_genes))
        
    p_diff_idx = random.sample(range(num_genes), n_diff_genes)
    diff_p_mass = sum(p_center[p_diff_idx])
    
    p_diff = np.random.beta(0.005, 1, n_diff_genes) + 1e-8
    #p_diffpart_pool = list(p_diffpart) + list(1-p_diffpart)
    #p_vec[p_diff_idx] = np.array(random.sample(p_diffpart_pool, n_diff_genes))
    
    #p_diff = np.random.uniform(0,1,n_diff_genes)
    p_vec[p_diff_idx] = p_diff/sum(p_diff)*diff_p_mass
    
    diri_p = p_vec/sum(p_vec)
    
    return list(precision * (diri_p.values))


def generate_alpha_arr(sample_component,precision,num_genes,n_diff_genes,dispersion=0.5):
    
    p_center = np.random.beta(0.005, 1, num_genes) + 1e-8
    p_center = p_center/sum(p_center)
    
    
    n_celltype = len(sample_component[0])
    alpha_list = []
    for ct in range(n_celltype):
        alpha_list.append(generate_alpha(precision[ct], num_genes, p_center, n_diff_genes,dispersion))
    
    alpha_arr = np.array(alpha_list)
    
    return alpha_arr
    

def get_w(d,logLS_m):
    
    m = [10**logLS_m[x] for x in d]
    w = m/np.sum(m)
    return w


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
    pmat = np.ndarray(pmat_shape, dtype=np.float, buffer=existing_shm.buf)
    
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
    np_array = np.ndarray(pmat_shape, dtype=np.float, buffer=shm.buf)
    np_array[:] = cell_pmat[:]  # Copy the original data into shared memory
    # del cell_pmat
    
    t0 = time.time()
    pool = mp.Pool()  
    compact_re = [pool.apply_async(sub_simulate_UMIs, (shm.name, pmat_shape, droplets[idx[i] : idx[i+1]])) for i in range(len(idx)-1)]
    pool.close()
    pool.join()
    print('multiprocessing: ', time.time()-t0)

    UMI_list = [item.get() for item in compact_re]
    UMI_flat_list = [item for sub in UMI_list for item in sub]

    return np.array(UMI_flat_list)



def simulate_Cells(cell_pmat, droplets):
    
    t0 = time.time()
    Cell_counts = []
    
    idx = 0
    for d in droplets.index: # for i in range(start,end):
        if idx % 1000 == 0:
            print(idx)
        idx += 1

        totm = droplets['cell_mRNA'][d]
        comp = droplets['component'][d]
        num = droplets['num_cells'][d]
        
        abs_counts = np.array([np.random.multinomial(totm[i], cell_pmat[comp[i]]) for i in range(num)]).sum(0)
        Cell_counts.append(list(abs_counts))   
    
    print('simulate cells:',time.time()-t0)
    
    return np.array(Cell_counts)



def QuasiGMMdemux_pureTypeHTO(droplets, celltype_anno):
    '''
    If cells are sorted before cell hashing, which means each HTO represents one certain cell type, 
    this function could classify droplets into singlets, two-cell-type-doublets and multiplets in the same way with GMM-Demux.
    
    Parameters
    ----------
    sample_composition : list of list
        configure cell type composition in each hto-tagged sample.
    celltype_anno : pd.Series
        cell type names.

    Returns
    -------
    GMMdemux : list
        classificaiton of droplets.

    '''
    sample_composition = [list(np.unique(x)) for x in droplets['HTO']]
    GMMdemux = []
    for x in sample_composition:
        if len(x) == 1:
            GMMdemux.append(celltype_anno[x[0]])
        elif len(x) == 2:
            GMMdemux.append('_'.join(['2mix']+[celltype_anno[c] for c in np.sort(x)]))
        else:
            GMMdemux.append('Multiplet')
            
    return GMMdemux


def sub_simSurfMarkers(adts_para,num_cells):
    return [[np.random.normal(x[0], x[1]) for x in adts_para.values] for i in range(num_cells)]
    

def simSurfMarkers_cells(ADTpara, sample_component):
    
    celltypes = list(ADTpara.keys())
    adt_list = []
    for hto in sample_component:
        for cidx, num_cells in enumerate(hto):
            ct = celltypes[cidx]
            adt_list.append(sub_simSurfMarkers(ADTpara[ct],num_cells))
    
    adt_df = pd.DataFrame(np.concatenate(adt_list), columns=ADTpara[ct].index)    
    return adt_df


def simSurfMarkers_droplets(droplets, cell_ADTs):
    
    dropletADTtmp = []
    for d in droplets.index:
        comp = droplets['component'][d]
        dropletADTtmp.append(cell_ADTs.values[comp,:].max(0))

    droplet_ADTs = pd.DataFrame(np.array(dropletADTtmp),columns=cell_ADTs.columns)
    return droplet_ADTs 
    

from sklearn.mixture import GaussianMixture
def QuasiCITEsort(droplet_ADTs, prior):
    
    pred_list = []
    for m in droplet_ADTs.columns:
        
        gmm = GaussianMixture(n_components=2).fit(droplet_ADTs[m].values.reshape(-1,1))
        pred_labels = gmm.predict(droplet_ADTs[m].values.reshape(-1,1))
        pred_symbol = ['+' if gmm.means_[v] > 0 else "-" for idx,v in enumerate(pred_labels)]

        pred_list.append(pred_symbol)

    pred_df = pd.DataFrame(np.transpose(np.array(pred_list)),columns=droplet_ADTs.columns)

    anno = []
    for d in pred_df.index:
        tmp = []
        for ct in prior:
            scores = 0
            for m in prior[ct]:
                if prior[ct][m] == pred_df.loc[d,m]:
                    scores += 1/len(prior[ct])
            if scores == 1:
                tmp.append(ct)
                
        if len(tmp) > 2:
            anno.append('Multiplet')
        else:
            anno.append('_'.join(tmp))
    
    return anno


# PART 2
# 定义好alpha_df, alpha_arr, logLS_m, logLS_std,提供celltype_names
# 
def generate_sc(sample_component,capture_rate, celltype_names, logLS_m, logLS_std, alpha_arr, gene_name):
#     sample_component = [[2000,2000,2000]]
    droplets = dropletFormation(sample_component,capture_rate)
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
    print(time.time() - t0)

    # data=np.array(UMI_mat)
    sc=pd.DataFrame(np.array(UMI_mat)).T
    sc=sc.set_index(gene_name).T
    sc['cell_num']=list(UMI_info['num_cells'])
    sc['cell_type']=list(UMI_info['cell_type'])
    sc['HTO']=UMI_info['HTO'].values
    sc=sc[sc['cell_num']==1]
    # sc=sc.drop['cell_type']
    sc['cell_type'] = sc['cell_type'].apply(lambda x: ', '.join(map(str, x)))
    sc['HTO'] =  sc['HTO'].apply(lambda lst: [x + 1 for x in lst])
    sc['HTO'] = sc['HTO'].apply(lambda x: "sample"+', '.join(map(str, x)))
    coded = sorted(list(set(sc['cell_type'])))
    # celltype_names= ['mNK','Bcell','CD4T']
    ct_map_dict = dict(zip(coded, celltype_names[0:len(coded)]))
    sc['cell_type'] = sc['cell_type'].map(ct_map_dict)

    sc_m=sc.iloc[:,0:len(gene_name)]
    sc_ct=sc.iloc[:,len(gene_name)]
    sc_m = sc.iloc[:,0:len(gene_name)]
    sc_info = sc[['cell_type']]
    sc_info['sampleID'] = sc['HTO']
    return sc_m, sc_info


def simulate_bulk_cell(cell_pmat, mRNA_acount):

    Cell_counts = []
    t = 0
    for d in range(0,len(mRNA_acount)):
        t += 1
        totm = mRNA_acount[d]
        abs_counts = np.random.multinomial(totm, cell_pmat[d])
        Cell_counts.append(list(abs_counts))

    return pd.DataFrame(Cell_counts)

# def generate_bulk(sample_component,  logLS_m, logLS_std, alpha_arr, gene_name,mu,sigma):
#     cell_pmat = cell_probV(alpha_arr, sample_component)
#     mRNA_acount = absoluteLS(sample_component, logLS_m, logLS_std)
#     cell=simulate_bulk_cell(cell_pmat, mRNA_acount)
#     cell=cell.T
#     cell=cell.set_index(gene_name)

#     start = 0
#     bulk = []
#     for sample in range(0,len(sample_component)):
#         cells = cell.iloc[:,start:start+sum(sample_component[sample])]
#         # print(cells)
#         start = start+sum(sample_component[sample])
#         cell_counts = cells.sum(1)
#         mRNA_pool = cells.index.repeat(cell_counts)
#         # 计算总数的 mRNA_capture_rate 抽样数量
#         mRNA_capture_rate = np.random.normal(mu, sigma)
#         while mRNA_capture_rate < 0:
#             mRNA_capture_rate = np.random.normal(mu, sigma)
#         sample_size = int(mRNA_capture_rate * len(mRNA_pool))
#         # 从池中按比例的概率抽样
#         sampled_genes = np.random.choice(mRNA_pool, size=sample_size, replace=False)

#         # 使用 Counter 对象直接统计抽样后的基因数量

#         sampled_bulk = {k: dict(Counter(sampled_genes))[k] for k in sorted(dict(Counter(sampled_genes)))}.values()
#         sampled_bulk_list = []
#         for d in sampled_bulk:
#             sampled_bulk_list.append(d)
#         bulk.append(sampled_bulk_list)
#     bulk=pd.DataFrame(bulk).T
#     sampleid_bulk = list(bulk.columns)
#     sampleid_bulk = [x + 1 for x in sampleid_bulk]
#     sampleid_bulk = ['sample'+str(item) for item in sampleid_bulk]
#     bulk.columns = sampleid_bulk
#     bulk = bulk.T
#     bulk.columns = gene_name
#     return bulk

def generate_bulk(sample_component,  logLS_m, logLS_std, alpha_arr, gene_name,mu,sigma):
    cell_pmat = cell_probV(alpha_arr, sample_component)
    mRNA_acount = absoluteLS(sample_component, logLS_m, logLS_std)
    cell=simulate_bulk_cell(cell_pmat, mRNA_acount)
    cell=cell.T
    cell=cell.set_index(gene_name)

    start = 0
    bulk = []
    for sample in range(0,len(sample_component)):
        cells = cell.iloc[:,start:start+sum(sample_component[sample])]
        # print(cells)
        start = start+sum(sample_component[sample])
        pool = pd.DataFrame(cells.sum(1),columns=['Count'])

        # 计算总数的 mRNA_capture_rate 抽样数量
        mRNA_capture_rate = np.random.normal(mu, sigma)
        while mRNA_capture_rate < 0:
            mRNA_capture_rate = np.random.normal(mu, sigma)
        sample_size = int(mRNA_capture_rate * pool.sum(0))

        sampled_ids = np.random.choice(pool.index, size=sample_size, p=pool['Count']/pool['Count'].sum(), replace=True)
        sampled_data = pd.DataFrame(sampled_ids, columns=['gene_name'])
        result = sampled_data['gene_name'].value_counts().reset_index()
        result = result.set_index('gene_name')
        bulk_single = list(result.reindex(cells.index)['count'])
        bulk.append(bulk_single)
    bulk=pd.DataFrame(bulk).T
    sampleid_bulk = list(bulk.columns)
    sampleid_bulk = [x + 1 for x in sampleid_bulk]
    sampleid_bulk = ['sample'+str(item) for item in sampleid_bulk]
    bulk.columns = sampleid_bulk
    bulk = bulk.T
    bulk.columns = gene_name
    return bulk


def get_ground_truth(sample_component):
    GT = []
    for i in range(0,len(sample_component)):
        sample_list = sample_component[i]
        prob_i = [x/sum(sample_list) for x in sample_list]
        GT.append(prob_i)
    return GT

def generate_data(file_name, sample_component_sc,sample_component_bulk,cell_capture , celltype_names, logLS_m, logLS_std, alpha_arr, gene_name,mu,sigma):
    sc_m,sc_info = generate_sc(sample_component_sc,cell_capture, celltype_names, logLS_m, logLS_std, alpha_arr, gene_name)
    bulk = generate_bulk(sample_component_bulk, logLS_m, logLS_std, alpha_arr, gene_name,mu,sigma)
    bulk_prob = pd.DataFrame(get_ground_truth(sample_component_bulk),index=bulk.index, columns=celltype_names).T
    
    # save data
    output = '/home/rui/Subtype_Deconvolution/simulated_data/'
    target_file = file_name
    target_path = os.path.join(os.path.dirname(output), target_file)

    if  not os.path.exists(target_path):
        os.makedirs(target_path)
        
    bulk.to_csv(os.path.join(target_path,'bulk'),header=True,index=True)
    sc_m.to_csv(os.path.join(target_path,'sc_m'),header=True,index=True)
    sc_info.to_csv(os.path.join(target_path,'sc_info'),header=True,index=True)
    bulk_prob.to_csv(os.path.join(target_path,'bulk_pro'),header=True,index=True)
    return bulk, sc_m, sc_info, bulk_prob
