import os, sys
import numpy as np
import pandas as pd
import random
from decimal import Decimal
from absl import app
from absl import flags

## zipf parameter 變化

FLAGS = flags.FLAGS
flags.DEFINE_string('dataset_name', 'test_dataset', 'dataset name')
flags.DEFINE_integer('req_kind', 1000, 'number of requests kind', lower_bound=10)
flags.DEFINE_integer('length', 3000, 'requests length', lower_bound=10)
flags.DEFINE_float('zipf_para', 1.3, 'zipf parameter')
flags.DEFINE_integer('change_para', None, 'popularity ranking change length')


def save_access_data(file_name, access_list):
    f = open('./dataset/req_trace/'+file_name+'.txt', "w")
    f.write(str(access_list))
    f.close()
    print('successfully save dataset to', './dataset/'+file_name+'.txt')


def gen_data(file_name, req_kind, length, zipf_para, change_para=3):
    files = np.arange(1, req_kind+1)
    # Random ranks. Note that it starts from 1.
    ranks = np.random.permutation(files)
    # print(ranks)

    # Distribution
    pdf = 1 / np.power(ranks, zipf_para)
    pdf /= np.sum(pdf)

    all_req = []
    for i in range(change_para):
        print(zipf_para)
        random.shuffle(ranks)
        pdf = 1 / np.power(ranks, zipf_para)
        pdf /= np.sum(pdf)
        # print(ranks)
        all_req.extend(list(np.random.choice(files, size=length, p=pdf)))
        zipf_para-=0.6
        

    ## Draw samples
    # print((all_req))

    ## Save Txt
    save_access_data(file_name, (all_req))
    ## Save csv for IL
    df_train = None
    df_valid = None
    hex_requestseq = [hex(i) for i in all_req]
    tmp_train = pd.DataFrame({'pc': hex(0), 'address': hex_requestseq})
    tmp_valid = pd.DataFrame({'pc': hex(0), 'address': hex_requestseq[len(all_req)//3*2:]})
    if (df_train is None) and (df_valid is None):
        df_train = tmp_train
        df_valid = tmp_valid
    else:
        df_train = pd.concat((df_train, tmp_train), axis = 0)
        df_valid = pd.concat((df_valid, tmp_valid), axis = 0)
    
    save_name = f'./dataset/req_trace/{file_name}'
    df_train.to_csv(f'{save_name}_train.csv', index = False, header = False)
    df_valid.to_csv(f'{save_name}_valid.csv', index = False, header = False)

def main(argv):
    

    file_name = f'{FLAGS.dataset_name}_{FLAGS.req_kind}_{FLAGS.length}_cps'
    req_kind = FLAGS.req_kind    ## req kind
    length = FLAGS.length//3  ## req num
    param = FLAGS.zipf_para
    change_para = FLAGS.change_para

    gen_data(file_name=file_name, req_kind=req_kind, length=length, zipf_para=param)



if __name__ == "__main__":
    app.run(main)
    


    