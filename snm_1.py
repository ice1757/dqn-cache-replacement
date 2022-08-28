from urllib import request
import numpy as np
import copy
import collections
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from absl import app
from absl import flags

FLAGS = flags.FLAGS
flags.DEFINE_string('dataset_name', 'snm_default', 'dataset name')
flags.DEFINE_integer('req_kind', 50, 'number of requests kind', lower_bound=1)
# flags.DEFINE_integer('length', 3000, 'requests length', lower_bound=1)

def save_access_data(file_name, access_list):
    f = open(file_name+'.txt', "w")
    f.write(str(access_list))
    f.close()
    print('successfully save dataset to', './dataset/req_trace/'+file_name+'.txt')

def gen_snm(file_name, req_kind):
    df_train = None
    df_valid = None

    T = 90  # shot duration
    M = req_kind  # 內容種類數量
    alpha = 0.8  # 冪律參數
    u_mean = 2 # 平均熱度 lambda

    zm = np.random.rand(M+1)  # zm用來計算um，數組長度為內容種類數量，從[0,1]中取值
    all_m_pop = zm**(-alpha)*u_mean*(1-alpha) # 內容m的平均熱度

    requestnum = all_m_pop * T #內容m 個別總共的請求數量
    wn = [] #單個內容請求到達的時間
    m_start_wn = {} #每個內容的請求到達的時間
    wn.append(0)
    T_interval = []
    for onem in range(1, M+1):
        T_interval = np.random.exponential(1/all_m_pop[onem],int(requestnum[onem]))
        for onetime in range(1,len(T_interval)+1):
            wn.append(onem*5+wn[onetime -1]+ T_interval[onetime - 1])
        m_start_wn[onem] = copy.copy(wn)
        wn = []
        wn.append(0)


    totallist = [] #將所有內容的請求時間序列放在一起
    for onekey in m_start_wn.keys():
        totallist += m_start_wn[onekey][1:]
    totallist.sort() #將其按從小到達排序

    num = 0
    requestseq = [-1]*len(totallist)#最後生成的請求序列
    for onem in range(1, M+1): #對每個內容做遍歷，依據其請求到達的時間節點決定請求序列中的位置
        num += 1
        wn = m_start_wn[onem]
        for onet in wn[1:]:
            requestseq[totallist.index(onet)] = onem

    ## for IL
    hex_requestseq = [hex(i) for i in requestseq] * 3
    tmp_train = pd.DataFrame({'pc': hex(0), 'address': hex_requestseq})
    tmp_valid = pd.DataFrame({'pc': hex(0), 'address': hex_requestseq[len(requestseq)//3*2:]})
    ## Save
    if (df_train is None) and (df_valid is None):
        df_train = tmp_train
        df_valid = tmp_valid
    else:
        df_train = pd.concat((df_train, tmp_train), axis = 0)
        df_valid = pd.concat((df_valid, tmp_valid), axis = 0)
    if len(requestseq) >= 9000 and  len(requestseq) <= 9999: #snm_100_l9659
        df_train.to_csv(f'{file_name}_l{len(hex_requestseq)}_train.csv', index = False, header = False)
        df_valid.to_csv(f'{file_name}_l{len(hex_requestseq)}_valid.csv', index = False, header = False)
        save_access_data(f'{file_name}_l{len(requestseq)}', requestseq)
        pass

    print(len(requestseq))
    return len(requestseq)

def main(argv):
    

    file_name = f'./dataset/req_trace/{FLAGS.dataset_name}_{FLAGS.req_kind}'
    req_kind = FLAGS.req_kind    ## req kind
    # length = FLAGS.length  ## req num

    cn = gen_snm(file_name=file_name, req_kind=req_kind)
    while(cn < 9000):
        gen_snm(file_name=file_name, req_kind=req_kind)
        print('Waiting for generating snm trace')



if __name__ == "__main__":
    app.run(main)