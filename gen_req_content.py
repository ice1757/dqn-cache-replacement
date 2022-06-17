import os, sys
import numpy as np
import pandas as pd
import random

def save_access_data(file_name, access_list):
    f = open('./dataset/req_data/'+file_name+'.txt', "w")
    f.write(str(access_list))
    f.close()
    print('successfully save dataset to', './dataset/req_data/'+file_name+'.txt')


if __name__ == "__main__":

    if len(sys.argv) != 4:
        print("Usage: %s <file_name> <num_resources> <max_content_size> " % sys.argv[0])
        exit(0)
    
    file_name = sys.argv[1]
    num_resource = int(sys.argv[2]) ## 幾種
    max_content_size = int(sys.argv[3])
    # param = float(sys.argv[4])

    size_ar = [i+1 for i in range(max_content_size)]
    # weight_ar = [10 for i in range(max_content_size-1)]
    # weight_ar.insert(0, 100-(10*(max_content_size-1)))
    weight_ar = [80, 10, 10]
    req_to_size = {}
    cn =[0, 0, 0]
    
    size = random.choices(size_ar, weights = weight_ar, k=num_resource)
    for i, j in enumerate(size):
        req_to_size[i+1] = j
    
    
    
    # for i in range(num_resource):
    #     size = random.choices(size_ar, weights = weight_ar, k=num_resource)
    #     if size == 1: cn[0]+=1
    #     elif size == 2: cn[1] +=1
    #     else: cn[2]+=1
    #     req_to_size[i+1] = size

    # print(weight_ar)
    # print(req_to_size)
    
    # Save
    save_access_data(file_name, req_to_size)