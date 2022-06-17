import os, sys
import numpy as np
import pandas as pd
import random
from decimal import Decimal
from absl import app
from absl import flags


FLAGS = flags.FLAGS
flags.DEFINE_string('dataset_name', 'test_dataset', 'dataset name')
flags.DEFINE_integer('req_kind', 1000, 'number of requests kind', lower_bound=1)
flags.DEFINE_integer('length', 3000, 'requests length', lower_bound=1)
flags.DEFINE_float('zipf_para', 1.3, 'zipf parameter')
flags.DEFINE_integer('change_rank', None, 'popularity ranking change length')


def save_access_data(file_name, access_list):
    f = open(file_name+'.txt', "w")
    f.write(str(access_list))
    f.close()
    print('successfully save dataset to', './dataset/'+file_name+'.txt')


def gen_data(file_name, req_kind, length, zipf_para, change_rank):
    files = np.arange(1, req_kind+1)
    
    # Random ranks. Note that it starts from 1.
    ranks = np.random.permutation(files)
    # print(ranks)

    # Distribution
    pdf = 1 / np.power(ranks, zipf_para)
    pdf /= np.sum(pdf)

    all_req = []
    if not change_rank: all_req = list(np.random.choice(files, size=length, p=pdf))
    else:
        for i in range(length):
            if i % change_rank == 0:
                random.shuffle(ranks)
                pdf = 1 / np.power(ranks, zipf_para)
                pdf /= np.sum(pdf)
                # print(ranks)
            all_req.append(list(np.random.choice(files, size=1, p=pdf))[0])
    # Draw samples
    # print((all_req))

    ## Save txt
    save_access_data(file_name, (all_req))

    ## Save csv
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
    
    
    df_train.to_csv(f'{file_name}_train.csv', index = False, header = False)
    df_valid.to_csv(f'{file_name}_valid.csv', index = False, header = False)
    

def main(argv):
    

    file_name = f'./dataset/req_trace/{FLAGS.dataset_name}_{FLAGS.req_kind}_{FLAGS.length}_cr{FLAGS.change_rank}'
    req_kind = FLAGS.req_kind    ## req kind
    length = FLAGS.length  ## req num
    param = FLAGS.zipf_para
    change_rank = FLAGS.change_rank

    gen_data(file_name=file_name, req_kind=req_kind, length=length, zipf_para=param, change_rank=change_rank)



if __name__ == "__main__":
    app.run(main)
    


    