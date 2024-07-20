import warnings
warnings.filterwarnings('ignore')
import pandas as pd
import numpy as np
import random
import os


def read_resutlt(data_path, file_name_path):

    import os
    import pandas as pd
    t=data_path
    img_list = os.listdir(t)

    path_list=[]
    for i in range(0, len(img_list)):
        img_full_path = os.path.join(data_path, img_list[i])  ##路径拼接
        if os.path.exists(img_full_path) and img_full_path.startswith(file_name_path): # 关键词筛选
            path_list.append(img_full_path)
        path_list.sort()

    list2=[]
    df1=pd.DataFrame()
    for i in range(0,len(path_list)):
        f=open(path_list[i])
        content=f.read().split(' ')
        list2.append(content)
    df=pd.DataFrame(list2,columns=['all'])
    df=df['all'].str.split('\n',expand=True)
    df1['result']=df[1]
    df1=df1['result'].str.split('\t',expand=True)
    return df1